#!/usr/bin/env python3
"""Seal CPU-ready sources, input registry, evidence and historical protections."""
import argparse
import ast
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from diagnostics import reference_routing_content_inputs_v1 as c
from diagnostics.cross_model_applicability_execution_v1 import framework_sources
from audit_reference_routing_content_inputs_v1 import audit


def sources():
    files=set((ROOT/'src/diagnostics').glob('reference_routing_content_*_v1.py'))
    files.update((ROOT/'scripts/review').glob('*reference_routing_content*v1.py'))
    todo=list(files)
    while todo:
        path=todo.pop()
        for node in ast.walk(ast.parse(path.read_text())):
            names=[]
            if isinstance(node,ast.ImportFrom):
                if node.module=='diagnostics':names=['diagnostics.'+a.name for a in node.names]
                elif node.module and (node.module.startswith('diagnostics.') or node.module.startswith('test_')):names=[node.module]
            elif isinstance(node,ast.Import):names=[x.name for x in node.names if x.name.startswith('diagnostics.')]
            for name in names:
                target=(ROOT/'src'/Path(*name.split('.'))).with_suffix('.py') if name.startswith('diagnostics.') else ROOT/'scripts/review'/(name+'.py')
                if target.is_file() and target not in files:files.add(target);todo.append(target)
    files.update(framework_sources())
    files.update(Path(x['path']) for x in c.read(c.OLD/'prepared-01/source-ledger.json')['files'])
    for folder in (c.PROTOCOL,c.DRAFT,c.ADOPTED):
        files.add(folder/'manifest.json')
        manifest=c.verify_manifest(folder/'manifest.json')
        for section in ('artifacts','sources'):
            files.update(Path(x['path']) if Path(x['path']).is_absolute() else ROOT/x['path'] for x in manifest.get(section,[]))
    for row in c.read(c.PREPARED/'historical-replay.json')['records']:
        for field in ('record','vector','states','trajectory'):
            if row[field]:c.verify(row[field]);files.add(Path(row[field]['path']))
    files.add(c.WORK/'checks/development-note.md')
    return sorted(files)


def seal(checks):
    p=c.PREPARED;c.require(not (p/'manifest.json').exists(),'Preparation already sealed')
    audit(p,sealed=False)
    evidence=[]
    for name in checks:
        path=c.WORK/'checks'/name;r=c.read(path)
        c.require(r['status']=='PASS' and not r.get('CUDA_initialized',False),'CPU evidence failed')
        evidence.append(c.info(path))
    c.require(len(evidence)==5,'Capture, runtime, analysis, pipeline and input evidence required')
    files=sources();pins=[c.info(path) for path in files]
    c.write(p/'source-ledger.json',{'files':pins,'CPU_checks':evidence})
    c.write(p/'CPU-READINESS.json',{'status':'CPU_READY_GPU_NOT_AUTHORIZED','material_bulk_adoption':c.info(c.ADOPTED/'manifest.json'),
        'checks':evidence,'real_model_forwards':0,'weights_loaded':False,'CUDA_initialized':False,
        'GPU_numerical_qualification':'pending_real_execution','development_parameters':'not_fitted_no_outputs',
        'confirmation_execution':'blocked_until_development_lock','GPU_execution_authorized':False})
    c.write(p/'manifest.json',{'schema_version':'reference-routing-content-prepared/v1','status':'CPU_READY_GPU_NOT_AUTHORIZED',
        'artifacts':[c.info(path) for path in sorted(p.rglob('*')) if path.is_file()],
        'sources':pins+evidence,'protected_selectors':c.read(c.DRAFT/'manifest.json')['protected_selectors'],
        'immutable_after_seal':True,'GPU_execution_authorized':False})
    return c.info(p/'manifest.json')


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser();parser.add_argument('--checks',nargs=5,required=True);args=parser.parse_args()
    print(json.dumps(seal(args.checks),indent=2))
