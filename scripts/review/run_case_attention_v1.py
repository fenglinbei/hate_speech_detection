#!/usr/bin/env python3
"""Case-attention CPU preparation; GPU work requires a separate explicit command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from diagnostics import case_attention_inputs_v1 as c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'validate', 'preview', 'seal', 'bind', 'run', '_worker', 'check', 'analyze'])
    p.add_argument('--prepared', type=Path, default=c.PREPARED)
    p.add_argument('--output', type=Path)
    p.add_argument('--run', type=Path)
    p.add_argument('--bound', type=Path)
    p.add_argument('--gpu', type=int)
    p.add_argument('--authorization-note')
    p.add_argument('--phase', choices=['engineering', 'full'], default='engineering')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--invocation')
    p.add_argument('--unsealed', action='store_true')
    args = p.parse_args()
    if args.command == 'prepare':
        result = c.prepare(args.prepared)
    elif args.command == 'validate':
        plan, _, requests = c.validate(args.prepared, sealed=not args.unsealed)
        result = {'status': 'CPU_valid', 'inputs': len(requests), 'GPU_qualified': False, 'budget': plan['budget']}
    elif args.command == 'seal':
        result = c.seal(args.prepared)
    elif args.command == 'preview':
        from diagnostics.case_attention_report_v1 import preview
        result = preview(args.prepared)
    else:
        from diagnostics import case_attention_runtime_v1 as rt
        if args.command == 'bind':
            c.require(args.gpu is not None and args.output and args.authorization_note, 'bind requires gpu, output, authorization-note')
            result = rt.bind(args.prepared, args.gpu, args.output, args.authorization_note)
        elif args.command in ('run', '_worker'):
            c.require(args.bound and args.run, 'run requires bound and run paths')
            if args.command == '_worker':
                c.require(args.invocation, 'Internal worker requires registered invocation')
                result = rt.worker(args.prepared, args.bound, args.run, args.invocation, args.phase)
            else:
                result = rt.supervise(args.prepared, args.bound, args.run, args.phase, args.resume)
        elif args.command == 'check':
            c.require(args.run, 'check requires run path')
            result = rt.check_run(args.prepared, args.run)
        else:
            c.require(args.run and args.output, 'analyze requires run and output')
            from diagnostics.case_attention_report_v1 import analyze
            result = analyze(args.prepared, args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
