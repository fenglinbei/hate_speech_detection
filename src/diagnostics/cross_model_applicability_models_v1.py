"""Pinned native model adapters. Imports are lazy; CPU checks never load a checkpoint."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types


def require(value, message):
    if not value: raise ValueError(message)


def load_glm_classes(directory):
    """Import only the explicitly pinned local implementation, without HF cache copies."""
    directory=Path(directory).resolve()
    package='cmad_local_glm_'+hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    if package+'.modeling_chatglm' not in sys.modules:
        module=types.ModuleType(package);module.__path__=[str(directory)];sys.modules[package]=module
        for stem in ['configuration_chatglm','modeling_chatglm']:
            name=package+'.'+stem
            spec=importlib.util.spec_from_file_location(name,directory/(stem+'.py'))
            loaded=importlib.util.module_from_spec(spec);sys.modules[name]=loaded;spec.loader.exec_module(loaded)
    return (sys.modules[package+'.configuration_chatglm'].ChatGLMConfig,
            sys.modules[package+'.modeling_chatglm'].ChatGLMForConditionalGeneration)


def configure_torch(cpu=False):
    import torch
    require(cpu or os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','CUBLAS determinism setting missing')
    torch.manual_seed(0);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    return torch


def device_map(profile, count):
    require(count in [1,2], 'This version supports one device or a contiguous two-device layer split only')
    if count==1: return {'':0}
    layers=profile['layers']
    if profile['architecture']=='qwen3':
        mapping={'model.embed_tokens':0,'model.rotary_emb':0,'model.norm':1,'lm_head':1}
        mapping.update({f'model.layers.{i}':min(i*count//layers,count-1) for i in range(layers)})
    else:
        mapping={'transformer.embedding':0,'transformer.rotary_pos_emb':0,
                 'transformer.encoder.final_layernorm':1,'transformer.output_layer':1}
        mapping.update({f'transformer.encoder.layers.{i}':min(i*count//layers,count-1) for i in range(layers)})
    return mapping


def prepared_input(ids, pad_token, padding='none', max_tokens=8192):
    ids=list(ids);n=len(ids)
    require(0<n<=max_tokens and all(isinstance(i,int) and i>=0 for i in ids),'invalid tokens or prohibited truncation')
    require(padding in ['none','left','right'],'unknown padding mode')
    extra=0 if padding=='none' else (n//16+1)*16-n
    require(n+extra<=max_tokens,'padded context exceeds bound')
    if padding=='left': seq,mask=[pad_token]*extra+ids,[0]*extra+[1]*n
    else: seq,mask=ids+[pad_token]*extra,[1]*n+[0]*extra
    position=[];seen=0
    for x in mask:
        seen+=x;position.append(max(0,seen-1))
    return {'input_ids':seq,'attention_mask':mask,'position_ids':position,
            'last_valid_index':max(i for i,v in enumerate(mask) if v),'valid_tokens':n,'tensor_tokens':len(seq)}


def forward_vector(model, architecture, ids, pad_token, padding='none', input_device=None):
    """Project only the final *valid* hidden state, including right-padded inputs.

    Both native causal-LM wrappers implement this same body-plus-LM-head operation.
    Avoid GLM return_last_logit on right padding, which selects a padding position.
    Batch size is exactly one; no labels, KV reuse or cross-prompt cache.
    """
    import torch
    p=prepared_input(ids,pad_token,padding)
    require(architecture in ['qwen3','chatglm'],'unsupported architecture')
    body=model.model if architecture=='qwen3' else model.transformer
    head=model.lm_head if architecture=='qwen3' else model.transformer.output_layer
    if input_device is None:
        embedding=model.get_input_embeddings() if architecture=='qwen3' else model.transformer.embedding.word_embeddings
        input_device=embedding.weight.device
    tensors={k:torch.tensor([p[k]],dtype=torch.long,device=input_device) for k in ['input_ids','attention_mask','position_ids']}
    require(not model.training,'model must be in eval mode')
    with torch.inference_mode():
        out=body(**tensors,use_cache=False,return_dict=True)
        require(out.past_key_values is None,'unexpected KV cache')
        hidden=out.last_hidden_state[:,p['last_valid_index']:p['last_valid_index']+1,:]
        logits=head(hidden)
        require(logits.shape==(1,1,model.config.vocab_size),'unexpected selected logits shape')
        require(logits.dtype==torch.float32 and bool(torch.isfinite(logits).all()),'logits must be finite native FP32')
        vector=logits[0,0].detach().cpu().numpy().copy()
    return p,vector


def load_checkpoint(profile, allocation):
    """GPU-only, called only after outer manifest, weight and idle-allocation checks."""
    torch=configure_torch()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==','.join(a['uuid'] for a in allocation),'visible GPU binding differs')
    require(torch.cuda.is_available() and torch.cuda.device_count()==len(allocation),'bound CUDA device count differs')
    for i,a in enumerate(allocation):
        observed=str(torch.cuda.get_device_properties(i).uuid).lower().removeprefix('gpu-')
        require(observed==a['uuid'].lower().removeprefix('gpu-'),'CUDA UUID differs from binding')
    from transformers import AutoConfig, AutoModelForCausalLM
    directory=profile['local_directory']
    if profile['architecture']=='chatglm':
        config_class,model_class=load_glm_classes(directory)
        config=config_class.from_pretrained(directory,local_files_only=True)
    else:
        config=AutoConfig.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        model_class=AutoModelForCausalLM
    config.torch_dtype=torch.float32;config._attn_implementation='eager';config.use_cache=False
    mapping=device_map(profile,len(allocation))
    model,loading=model_class.from_pretrained(directory,config=config,local_files_only=True,
        use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager',
        device_map=mapping,low_cpu_mem_usage=True,output_loading_info=True)
    require(not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),
            'checkpoint loading inventory mismatch')
    expected=set(json.loads((Path(directory)/'model.safetensors.index.json').read_text())['weight_map'])
    require(set(model.state_dict())==expected,'loaded state inventory differs from pinned checkpoint')
    model.eval()
    require(model.config._attn_implementation=='eager','attention implementation differs')
    layout=[]
    for kind,items in [('parameter',model.named_parameters()),('buffer',model.named_buffers())]:
        for name,tensor in items:
            require(tensor.device.type=='cuda' and tensor.device.index<len(allocation),'CPU/disk/meta offload is prohibited')
            require(not tensor.is_floating_point() or tensor.dtype==torch.float32,'non-FP32 model tensor')
            layout.append({'kind':kind,'name':name,'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device)})
    devices=[{'uuid':a['uuid'],'name':torch.cuda.get_device_properties(i).name,
              'memory_bytes':torch.cuda.get_device_properties(i).total_memory,
              'capability':list(torch.cuda.get_device_capability(i))} for i,a in enumerate(allocation)]
    identity={'devices':devices,'device_map':mapping,'attention':'eager','compute_dtype':'float32',
              'use_cache':False,'batch_size':1,'tf32':False,'deterministic_algorithms':True,
              'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
              'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,'loading_info':loading,
              'tensor_layout_sha256':hashlib.sha256(json.dumps(layout,sort_keys=True).encode()).hexdigest()}
    return model,identity
