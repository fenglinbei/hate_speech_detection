#!/usr/bin/env python3
"""Resume the unchanged scientific executor under the new polling authorization."""
from pathlib import Path
import argparse
import json
import os
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from diagnostics.q01_module_poll_v2 import verify_amendment, validate_config, epoch
from diagnostics.q01_module_package import read_json
from diagnostics.general_model_evidence_evaluation import require, file_sha

# Multiprocessing imports this entry point before entering the original worker.
if __name__ == '__mp_main__':
    expected = os.environ.get('Q01_MODULE_POLL_AMENDMENT_SHA256')
    require(expected is not None, 'spawned worker lacks scheduling source binding')
    verify_amendment(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    require(file_sha(args.config) == args.config_sha256, 'polling configuration changed after dispatch')
    config = read_json(args.config)
    validate_config(config)
    verify_amendment(config['amendment_sha256'])
    from scripts.review.run_q01_module_refinement import preparation_check
    from diagnostics.q01_module_execution import execute
    require(file_sha(Path(config['run']) / 'run_manifest.json') == config['paused_run_manifest_sha256'],
            'run changed since this polling authorization was prepared')
    preparation_check(config['plan'], config['run'])
    os.environ['Q01_MODULE_POLL_AMENDMENT_SHA256'] = config['amendment_sha256']

    def stop_owned_run(signum, frame):
        raise KeyboardInterrupt('cancelled authorized run; preserve complete pairs and release workers')

    signal.signal(signal.SIGTERM, stop_owned_run)
    result = execute(config['plan'], config['run'], config['device_indices'],
                     stop_epoch=epoch(config.get('checkpoint_at')), through_pass=None)
    print(json.dumps({'status': result['status'], 'completed_passes': result['completed_passes']}))


if __name__ == '__main__':
    main()
