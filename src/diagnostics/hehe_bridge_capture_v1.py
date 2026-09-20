"""Read-only pre-answer trajectories around an explicitly isolated query patch."""
from __future__ import annotations
import numpy as np
from diagnostics import hehe_focal_patch_capture_v1 as patcher
from diagnostics.case_attention_inputs_v1 import require
from diagnostics.cross_model_applicability_models_v1 import prepared_input
from diagnostics.hehe_presentation_inputs_v1 import MECHANISM_RULES

array_sha = patcher.array_sha
donor_rows = patcher.donor_rows
ARRAYS = ['states','normalized','lens_logits','branches']


def forward(model, request, pad_token, *, positions=(), observe=False, layer=None,
            replacement=None, padding='none', prefix_length=None, append_tokens=(), trajectory=False):
    import torch
    require(not trajectory or (prefix_length is None and not append_tokens), 'Trajectory only for complete scoring input')
    if not trajectory:
        v,s,proof = patcher.forward(model,request,pad_token,positions=positions,observe=observe,
            layer=layer,replacement=replacement,padding=padding,prefix_length=prefix_length,append_tokens=append_tokens)
        return v,s,proof,None
    pos = request['roles']['pre_answer']
    require(len(pos)==1 and pos[0]==request['prompt_tokens']-1, 'Expected final prompt position')
    require(pos[0] not in positions or layer is None, 'Readout cannot be patched')
    p=prepared_input(request['input_ids'],pad_token,padding)
    valid=[i for i,x in enumerate(p['attention_mask']) if x]; selected=valid[pos[0]]
    ids=[request['candidate_tokens']['有'],request['candidate_tokens']['无']]
    weight=model.lm_head.weight[ids].detach();norm=model.model.norm
    cache={};values={};handles=[]

    def row(t):
        require(t.dtype==torch.float32 and t.shape==(1,len(p['input_ids']),model.config.hidden_size), 'Readout state geometry changed')
        return t[0,selected].detach()

    def pre(li):
        def hook(module,args,kwargs):
            require(li not in cache, 'Repeated block')
            cache[li]={'pre':row(kwargs.get('hidden_states',args[0] if args else None))}
        return hook

    def mid(li):
        def hook(module,args): cache[li]['mid']=row(args[0])
        return hook

    def branch(li,name):
        def hook(module,args,out): cache[li][name]=row(out[0] if name=='attention' else out)
        return hook

    def post(li):
        def hook(module,args,out):
            data=cache.pop(li);data['post']=row(out)
            states=torch.stack([data[n] for n in ['pre','mid','post']])
            normalized=norm(states)
            lens=torch.nn.functional.linear(normalized,weight)
            tensors={'states':states,'normalized':normalized,'lens_logits':lens,
                     'branches':torch.stack([data['attention'],data['mlp']])}
            values[li]={k:v.cpu().numpy().copy() for k,v in tensors.items()}
        return hook

    try:
        for li,block in enumerate(model.model.layers):
            handles += [block.register_forward_pre_hook(pre(li),with_kwargs=True),
                block.post_attention_layernorm.register_forward_pre_hook(mid(li)),
                block.self_attn.register_forward_hook(branch(li,'attention')),
                block.mlp.register_forward_hook(branch(li,'mlp')),block.register_forward_hook(post(li))]
        # The patcher registers its sole mutation after our collectors. Readout
        # position is disjoint, so the same-layer collected row stays exact;
        # subsequent layers observe the actual patched residual stream.
        v,s,proof=patcher.forward(model,request,pad_token,positions=positions,observe=observe,
                                 layer=layer,replacement=replacement,padding=padding)
    finally:
        for h in reversed(handles): h.remove()
    require(not cache and set(values)==set(range(len(model.model.layers))), 'Incomplete trajectory')
    result={name:np.stack([values[i][name] for i in range(len(values))]) for name in ARRAYS}
    result.update(position=np.asarray(pos,dtype=np.int64),candidate_ids=np.asarray(ids,dtype=np.int64),
        label_weights=weight.cpu().numpy().copy(),norm_weight=norm.weight.detach().cpu().numpy().copy(),
        norm_eps=np.asarray([norm.variance_epsilon],dtype=np.float64))
    return v,s,proof,result


def scaled_error(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    require(a.shape==b.shape, 'Trajectory comparison shape changed')
    absolute=float(np.abs(a-b).max(initial=0.))
    return {'absolute':absolute,'scaled':absolute/max(1.,float(np.abs(a).max(initial=0)),float(np.abs(b).max(initial=0)))}


def validate_trajectory(t,req,profile,vector):
    L,D=profile['layers'],profile['hidden_size']
    shapes={'states':(L,3,D),'normalized':(L,3,D),'lens_logits':(L,3,2),
            'branches':(L,2,D),'label_weights':(2,D),'norm_weight':(D,)}
    require(t is not None, 'Missing trajectory')
    for name,shape in shapes.items():
        require(t[name].shape==shape and t[name].dtype==np.float32 and np.isfinite(t[name]).all(), 'Invalid trajectory '+name)
    require(t['position'].tolist()==req['roles']['pre_answer'], 'Readout position changed')
    require(t['candidate_ids'].tolist()==[profile['candidate_tokens']['有'],profile['candidate_tokens']['无']], 'Readout tokens changed')
    require(t['norm_eps'].tolist()==[profile['rms_norm_eps']], 'RMS epsilon changed')
    h=t['states'].astype(np.float64);b=t['branches'].astype(np.float64)
    norm=h/np.sqrt(np.mean(h*h,axis=-1,keepdims=True)+t['norm_eps'][0])*t['norm_weight']
    projected=t['normalized'].astype(np.float64)@t['label_weights'].astype(np.float64).T
    checks={'attention_residual':scaled_error(h[:,0]+b[:,0],h[:,1]),
        'mlp_residual':scaled_error(h[:,1]+b[:,1],h[:,2]),
        'continuity':scaled_error(h[:-1,2],h[1:,0]),
        'normalization':scaled_error(norm,t['normalized']),
        'projection':scaled_error(projected,t['lens_logits'])}
    require(checks['continuity']['absolute']==0, 'Residual continuity differs')
    for name,val in checks.items():
        require(val['scaled']<=MECHANISM_RULES['reconstruction_scaled_max_cap'], 'Trajectory reconstruction failed: '+name)
    error=max(checks['projection']['absolute'],float(abs(t['lens_logits'][-1,2].astype(np.float64)-vector[t['candidate_ids']]).max()))
    require(error<=MECHANISM_RULES['projection_absolute_cap'], 'Projection absolute gate failed')
    return {'checks':checks,'projection_absolute_error':error}


def compare_trajectories(a,b,kind):
    exact=kind in ['repeat','reverse','self','production_replay']
    rows={k:scaled_error(a[k],b[k]) for k in ARRAYS}
    for name,val in rows.items():
        require(val['scaled']<=(0 if exact else MECHANISM_RULES['padding_prefix_scaled_max_cap']), 'Trajectory invariance failed: '+kind+'/'+name)
    require(rows['lens_logits']['absolute']<=(0 if exact else MECHANISM_RULES['projection_absolute_cap']), 'Trajectory probe invariance failed')
    for name in ['candidate_ids','label_weights','norm_weight','norm_eps','position']:
        require(np.array_equal(a[name],b[name]), 'Trajectory static axes/weights changed')
    return {'kind':kind,'arrays':rows}


def structural_guard(patched,native,layer):
    for name in ARRAYS:
        require(np.array_equal(patched[name][:layer+1],native[name][:layer+1]), 'Pre-answer trajectory changed before cross-position propagation: '+name)
    if layer+1<len(native['states']):
        require(np.array_equal(patched['states'][layer+1,0],native['states'][layer+1,0]), 'Next-layer entry changed before attention')
    return {'through_layer':layer,'all_early_arrays_exact':True,'next_layer_entry_exact':True}
