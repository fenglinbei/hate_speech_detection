"""Separately pinned scheduling amendment; scientific executor remains unchanged."""
from datetime import datetime
from pathlib import Path

from diagnostics.general_model_evidence_evaluation import require, file_sha
from diagnostics.q01_module_package import ROOT, WORK, FREEZE, read_json

AMENDMENT = WORK / 'idle-resume-02/manifest.json'
PARENT_MANIFEST = '63683f4ebedefbf710f0cb3bfd8aab2ef2b865fd6a0abf779eb84c3a212c05c4'


def epoch(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, 'deadline must include timezone')
    return parsed.timestamp()


def verify_amendment(expected_sha256=None):
    if expected_sha256 is not None:
        require(file_sha(AMENDMENT) == expected_sha256, 'scheduling amendment hash changed')
    manifest = read_json(AMENDMENT)
    require(manifest['status'] == 'frozen' and manifest['schema_version'] == 'q01-module-idle-resume/v2',
            'scheduling amendment is not frozen')
    require(file_sha(FREEZE / 'manifest.json') == PARENT_MANIFEST == manifest['parent_manifest_sha256'],
            'scientific freeze changed')
    for name, expected in manifest['source_files'].items():
        require(file_sha(ROOT / name) == expected, 'scheduling source changed: ' + name)
    return manifest


def validate_config(config):
    require(config['schema_version'] == 'q01-module-idle-config/v2', 'wrong scheduler config')
    require(config['poll_interval_seconds'] == 900, 'this authorization requires 15-minute polling')
    require(config['device_indices'] == [0, 1, 2, 3], 'resume requires the original four GPUs')
    require(Path(config['plan']).resolve() == FREEZE and Path(config['run']).resolve() == WORK / 'run-01',
            'resume scope differs from the existing frozen run')
    require(config['phase'] == 'full', 'continue engineering gates then the complete scientific schedule')
    first = epoch(config['first_check_at'])
    stop = epoch(config.get('checkpoint_at'))
    release = epoch(config.get('release_deadline'))
    end = epoch(config.get('user_window_end'))
    if end is None:
        require(config.get('run_until_complete') is True and stop is None and release is None,
                'no-deadline policy must explicitly mean run until completion')
    else:
        require(not config.get('run_until_complete') and stop is not None and release is not None,
                'finite window must specify checkpoint and release deadlines')
        require(first < stop < release and release + 55 < end, 'invalid release deadline ordering')
    return first, stop, release, end


def decision(config, timestamp):
    first, stop, _, _ = validate_config(config)
    if stop is not None and timestamp >= stop:
        return {'action': 'window_closed', 'next_check_epoch': None}
    if timestamp < first:
        return {'action': 'wait', 'next_check_epoch': first}
    next_check = first + (int((timestamp - first) // 900) + 1) * 900
    return {'action': 'check', 'next_check_epoch': min(next_check, stop) if stop is not None else next_check}


def command(config_path, config):
    validate_config(config)
    require(read_json(config_path) == config, 'polling configuration changed before launch')
    return [str(ROOT / '.conda/stage1-p0/bin/python'), str(ROOT / 'scripts/review/run_q01_module_polled_v2.py'),
            '--config', str(Path(config_path).resolve()), '--config-sha256', file_sha(config_path)]
