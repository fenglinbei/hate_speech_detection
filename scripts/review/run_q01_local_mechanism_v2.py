#!/usr/bin/env python3
"""Versioned JSON receipt fix; same frozen scientific plan, worker, and checkpoints."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import signal
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.q01_mechanism_resume_v2 import verify_amendment, verify_resume
from diagnostics.q01_mechanism_package import load_frozen, FREEZE, WORK
from diagnostics.general_model_evidence_evaluation import require

# Also executes when multiprocessing imports this script in each spawned worker.
verify_amendment()
from diagnostics.q01_mechanism_execution_v2 import execute, check_run, analyze


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('validate', 'resume-check', 'run', 'check', 'analyze'))
    p.add_argument('--plan', type=Path, default=FREEZE)
    p.add_argument('--run', type=Path, default=WORK / 'run-01')
    p.add_argument('--output', type=Path, default=WORK / 'results-01')
    p.add_argument('--gpus', type=int, nargs='+')
    p.add_argument('--stop-at')
    args = p.parse_args()
    require(args.plan.resolve() == FREEZE and args.run.resolve() == WORK / 'run-01', 'amendment scope differs')
    if args.command == 'validate':
        plan, _, _, _ = load_frozen(args.plan)
        result = {'status': 'validated', 'plan_id': plan['plan_id'], 'gpu_forward_executed': False}
    elif args.command == 'resume-check':
        result = verify_resume(args.plan, args.run)
    elif args.command == 'run':
        require(args.gpus == [0, 1, 2, 3] and args.stop_at, 'resume requires original four cards and explicit deadline')
        stop = datetime.fromisoformat(args.stop_at)
        require(stop.tzinfo is not None, 'deadline requires timezone')
        def stop_owned_run(signum, frame):
            raise KeyboardInterrupt('termination: preserve checkpoint and release owned workers')
        signal.signal(signal.SIGTERM, stop_owned_run)
        result = execute(args.plan, args.run, args.gpus, stop_epoch=stop.timestamp())
    elif args.command == 'check':
        check_run(args.plan, args.run)
        result = {'status': 'verified', 'gpu_forward_executed': False}
    else:
        result = analyze(args.plan, args.run, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
