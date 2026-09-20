"""Fixed upstream query patch plus ordered pre-answer branch restorations.

The restored attention tensor is the output after o_proj, before residual
addition. MLP restoration likewise precedes residual addition. Hooks collect
the installed tensor, not the discarded counterfactual branch output.
"""
from __future__ import annotations
import numpy as np
from diagnostics import hehe_bridge_capture_v1 as bridge
from diagnostics.case_attention_inputs_v1 import require
from diagnostics.cross_model_applicability_models_v1 import prepared_input

array_sha = bridge.array_sha
donor_rows = bridge.donor_rows
validate_trajectory = bridge.validate_trajectory
compare_trajectories = bridge.compare_trajectories
structural_guard = bridge.structural_guard
ARRAYS = bridge.ARRAYS


def forward(model, request, pad_token, *, restoration=None, **kwargs):
    if restoration is None:
        v,s,p,t = bridge.forward(model,request,pad_token,**kwargs)
        if p is not None: p = dict(p,restoration=None)
        return v,s,p,t
    import torch
    require(isinstance(restoration,list) and 1 <= len(restoration) <= 2, 'Expected one or two ordered restorations')
    keys=[(spec['layer'],0 if spec['branch']=='attention' else 1) for spec in restoration]
    require(keys==sorted(set(keys)), 'Restorations must be unique and ordered')
    require(kwargs.get('prefix_length') is None and not kwargs.get('observe',False), 'Restoration is not donor capture')
    ids=request['input_ids']+list(kwargs.get('append_tokens',()))
    padded=prepared_input(ids,pad_token,kwargs.get('padding','none'))
    valid=[i for i,x in enumerate(padded['attention_mask']) if x]
    # Validate every spec before installing any hook.
    for spec in restoration:
        li,name,pos,value=spec['layer'],spec['branch'],spec['position'],spec['value']
        require(kwargs.get('layer') is not None and kwargs['layer'] < li < len(model.model.layers), 'Restoration must follow upstream patch')
        require(name in ['attention','mlp'] and pos == request['roles']['pre_answer'][0], 'Restore only registered pre-answer branch')
        require(value.dtype==np.float32 and value.shape==(model.config.hidden_size,) and np.isfinite(value).all(), 'Invalid restoration vector')
    proofs=[None]*len(restoration)
    def installer(index,spec):
        li,name,pos,value=spec['layer'],spec['branch'],spec['position'],spec['value']
        mapped=valid[pos]
        def install(module,args,out):
            require(proofs[index] is None,'Restoration hook ran twice')
            require(all(p is not None for p in proofs[:index]),'Restoration hooks ran out of order')
            require(name!='attention' or isinstance(out,tuple),'Attention output contract changed')
            tensor=out[0] if name=='attention' else out
            require(isinstance(tensor,torch.Tensor) and tensor.dtype==torch.float32 and
                    tensor.shape==(1,len(padded['input_ids']),model.config.hidden_size),'Restoration tensor geometry changed')
            original=tensor[0,mapped].detach().cpu().numpy().copy()
            installed=torch.as_tensor(value,device=tensor.device,dtype=tensor.dtype)
            changed=tensor.clone();changed[0,mapped]=installed
            rows=torch.nonzero(torch.any(changed!=tensor,dim=-1)[0],as_tuple=False).flatten().tolist()
            require(set(rows)<={mapped} and torch.equal(changed[0,mapped],installed),'Restoration changed unregistered rows')
            require(np.array_equal(tensor[0,mapped].detach().cpu().numpy(),original),'Original branch mutated in place')
            proofs[index]={'layer':li,'branch':name,'position':pos,'padded_position':mapped,
                'replacement_sha256':array_sha(value),'before_sha256':array_sha(original),
                'replacement_l2':float(np.linalg.norm(value.astype(np.float64))),
                'before_l2':float(np.linalg.norm(original.astype(np.float64))),
                'difference_l2':float(np.linalg.norm(value.astype(np.float64)-original)),
                'changed_rows_in_unpadded_coordinates':[valid.index(i) for i in rows],
                'outside_rows_exact':True,'replacement_exact':True,'native_output_unmodified':True,
                'other_attention_outputs_preserved':name=='attention'}
            return (changed,*out[1:]) if name=='attention' else changed
        return install
    handles=[]
    try:
        # Both mutations precede read-only collectors, including at the later site.
        for index,spec in enumerate(restoration):
            block=model.model.layers[spec['layer']]
            module=block.self_attn if spec['branch']=='attention' else block.mlp
            handles.append(module.register_forward_hook(installer(index,spec)))
        v,s,p,t=bridge.forward(model,request,pad_token,**kwargs)
    finally:
        for handle in reversed(handles):handle.remove()
    require(all(x is not None for x in proofs) and p is not None,'Missing intervention proof')
    return v,s,dict(p,restoration=proofs),t


def restoration_guard(restored,upstream,native,spec):
    """Exact causal boundary, installed vector, and unchanged static axes."""
    li=spec['layer']; bi=0 if spec['branch']=='attention' else 1
    for name in ARRAYS:
        require(np.array_equal(restored[name][:li],upstream[name][:li]),'Restoration changed earlier layers: '+name)
    # Attention restoration can change mid/post; MLP restoration only post.
    for name in ['states','normalized','lens_logits']:
        require(np.array_equal(restored[name][li,:bi+1],upstream[name][li,:bi+1]),'Restoration changed earlier same-layer sites')
    if bi==1:
        require(np.array_equal(restored['branches'][li,0],upstream['branches'][li,0]),'MLP restoration changed preceding attention')
    source=upstream if spec['source']=='upstream' else native
    require(np.array_equal(restored['branches'][li,bi],source['branches'][li,bi]),'Collected branch is not the installed source')
    return {'layer':li,'branch':spec['branch'],'all_preceding_arrays_exact':True,'installed_branch_exact':True}
