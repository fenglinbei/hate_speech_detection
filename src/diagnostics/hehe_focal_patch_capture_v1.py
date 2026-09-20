"""Explicit full-vector block-output interventions at native query positions.

No reference labels are read here. Donors are measured in the current run.
This module never calls a GPU inventory API or loads a checkpoint on import.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import numpy as np

from diagnostics.case_attention_inputs_v1 import require
from diagnostics import case_attention_capture_v1 as native
from diagnostics.cross_model_applicability_models_v1 import prepared_input


def array_sha(value):
    a=np.asarray(value)
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def forward(model, request, pad_token, *, positions=(), observe=False,
            layer=None, replacement=None, padding='none', prefix_length=None,
            append_tokens=()):
    """Capture all block outputs OR patch one layer, then read full vocabulary.

    `positions` always uses the unpadded prompt coordinate system. A generation
    continuation repeats the same prompt intervention; no KV cache is reused.
    Prefix collection is diagnostic and does not return an answer prediction.
    """
    import torch
    patch=layer is not None
    require(not (observe and patch),'Read-only capture and intervention are separate calls')
    require(patch == (replacement is not None),'Incomplete intervention')
    require(not (prefix_length is not None and append_tokens),'Prefix plus continuation is undefined')
    req=deepcopy(request)
    original_length=request['prompt_tokens']
    require(len(request['input_ids'])==original_length,'Expect one complete native prompt')
    if prefix_length is not None:
        require(0<prefix_length<=original_length,'Bad true prefix')
        req['input_ids']=request['input_ids'][:prefix_length]
    else:
        require(all(isinstance(t,int) and 0<=t<model.config.vocab_size for t in append_tokens),'Bad continuation token')
        req['input_ids']=request['input_ids']+list(append_tokens)
    ps=list(positions)
    require(ps==sorted(set(ps)) and all(isinstance(p,int) and 0<=p<len(req['input_ids']) for p in ps),'Bad observer positions')
    require(not (patch or observe) or ps,'Explicit positions required')
    if patch:
        require(0<=layer<len(model.model.layers),'Bad patch layer')
        require(set(ps)<=set(request['roles']['query_all']) and max(ps)<original_length-1,'Patch only query positions before readout')
        require(replacement.dtype==np.float32 and replacement.shape==(len(ps),model.config.hidden_size)
                and np.isfinite(replacement).all(),'Expected finite native FP32 donor rows')
    p=prepared_input(req['input_ids'],pad_token,padding)
    valid=[i for i,v in enumerate(p['attention_mask']) if v]
    mapped=[valid[i] for i in ps]
    states={};handles=[];proofs=[]

    def collector(li):
        def hook(module,args,output):
            require(isinstance(output,torch.Tensor) and output.dtype==torch.float32,'Native decoder output changed')
            require(output.shape==(1,len(p['input_ids']),model.config.hidden_size),'Decoder output geometry changed')
            require(li not in states,'Observer ran twice')
            states[li]=output[0,mapped].detach().cpu().numpy().copy()
        return hook

    def intervention(module,args,output):
        require(not proofs,'Patch hook ran twice')
        require(isinstance(output,torch.Tensor) and output.dtype==torch.float32,'Patch expects native FP32 block output')
        require(output.shape==(1,len(p['input_ids']),model.config.hidden_size),'Patch output geometry changed')
        rows=torch.as_tensor(replacement,dtype=output.dtype,device=output.device)
        before=output[0,mapped].detach().cpu().numpy().copy()
        changed=output.clone()
        changed[0,mapped]=rows
        require(torch.equal(changed[0,mapped],rows),'Donor was not installed exactly')
        changed_rows=torch.nonzero(torch.any(changed!=output,dim=-1)[0],as_tuple=False).flatten().tolist()
        require(set(changed_rows)<=set(mapped),'Intervention modified unselected rows')
        require(np.array_equal(output[0,mapped].detach().cpu().numpy(),before),'Native output mutated in place')
        proofs.append({'layer':int(layer),'positions':ps,'padded_positions':mapped,
                       'donor_sha256':array_sha(replacement),'recipient_before_sha256':array_sha(before),
                       'changed_rows_in_unpadded_coordinates':[valid.index(i) for i in changed_rows],
                       'outside_rows_exact':True,'donor_rows_exact':True,'native_output_unmodified':True})
        return changed

    try:
        if observe:
            for li,block in enumerate(model.model.layers):handles.append(block.register_forward_hook(collector(li)))
        if patch:handles.append(model.model.layers[layer].register_forward_hook(intervention))
        vector,_,_=native.forward(model,req,pad_token,padding=padding,capture=False)
    finally:
        for h in reversed(handles):h.remove()
    require(vector.dtype==np.float32 and vector.shape==(model.config.vocab_size,) and np.isfinite(vector).all(),'Bad final logit vector')
    if observe:
        require(set(states)==set(range(len(model.model.layers))),'Incomplete donor state collection')
        values=np.stack([states[li] for li in range(len(states))])
        require(values.dtype==np.float32 and np.isfinite(values).all(),'Bad captured states')
    else:values=None
    require(len(proofs)==int(patch),'Intervention execution count differs')
    return vector,values,proofs[0] if patch else None


def donor_rows(states, observed_positions, layer, positions):
    require(states.dtype==np.float32 and states.ndim==3,'Bad donor tensor')
    require(states.shape[1]==len(observed_positions),'Donor position axis mismatch')
    indices=[list(observed_positions).index(p) for p in positions]
    require(len(set(indices))==len(indices),'Duplicate donor positions')
    return states[layer,indices].copy()
