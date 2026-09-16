#!/usr/bin/env python3
"""Validate, explicitly run, check or analyze the separate Q01 mechanism freeze."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import signal
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.q01_mechanism_package import FREEZE, WORK, load_frozen
from diagnostics.q01_mechanism_execution import execute, check_run, analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('validate', 'run', 'check', 'analyze'))
    p.add_argument('--plan', type=Path, default=FREEZE)
    p.add_argument('--run', type=Path, default=WORK / 'run-01')
    p.add_argument('--output', type=Path, default=WORK / 'results-01')
    p.add_argument('--gpus', type=int, nargs='+')
    p.add_argument('--stop-at', help='Timezone-qualified ISO timestamp; checkpoint and exit after the current request.')
    p.add_argument('--through-pass', help='Pause after this accepted pass; use engineering-replica for tool-only acceptance.')
    args = p.parse_args()
    if args.command == 'validate':
        plan, contexts, _, requests = load_frozen(args.plan)
        result = {'status': 'validated', 'plan_id': plan['plan_id'], 'budget': plan['budget'], 'gpu_forward_executed': False}
    elif args.command == 'run':
        if not args.gpus: p.error('run requires an explicit physical --gpus allocation')
        stop = datetime.fromisoformat(args.stop_at) if args.stop_at else None
        if stop is not None and stop.tzinfo is None: p.error('--stop-at must include its UTC offset')
        def stop_owned_run(signum, frame):
            raise KeyboardInterrupt('received termination signal; preserving checkpoint and releasing owned workers')
        signal.signal(signal.SIGTERM, stop_owned_run)
        result = execute(args.plan, args.run, args.gpus, stop_epoch=stop.timestamp() if stop else None, through_pass=args.through_pass)
    elif args.command == 'check':
        check_run(args.plan, args.run); result = {'status': 'verified', 'gpu_forward_executed': False}
    else: result = analyze(args.plan, args.run, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
