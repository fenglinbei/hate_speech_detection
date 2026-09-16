#!/usr/bin/env python3
"""Read-only remote deployment receipt; never returns credentials or human text."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def service(name):
    keys = ['MainPID', 'ExecMainStartTimestamp', 'NRestarts', 'ActiveState', 'SubState']
    raw = subprocess.check_output(['systemctl', 'show', name, *['--property=' + k for k in keys]], text=True)
    return dict(line.split('=', 1) for line in raw.splitlines())

def https(host):
    result = subprocess.run(['curl', '--silent', '--show-error', '--fail', '--noproxy', '*', '--resolve', host + ':443:127.0.0.1', 'https://' + host + '/'], capture_output=True, timeout=25)
    assert result.returncode == 0, host + ': HTTPS failed'
    return {'certificate_verified': True, 'body_sha256': sha(result.stdout), 'bytes': len(result.stdout)}

def record(path):
    path = Path(path)
    if not path.exists():
        return None
    raw = path.read_bytes()
    value = json.loads(raw)
    return {'sha256': sha(raw), 'revision': value['revision'], 'reviewer_id': value['reviewer_id'], 'case_count': len(value['records']), 'confirmed_count': sum(x['status'] == 'confirmed' for x in value['records'].values()), 'bytes': len(raw)}

pdf_files = ['/etc/nginx/conf.d/pdf-translate-reader.conf', '/etc/systemd/system/pdf-translate-reader.service']
pdf_files += [str(p) for p in sorted(Path('/var/www/pdf-translate-reader').rglob('*')) if p.is_file()]
print(json.dumps({'observed_at': datetime.now(timezone.utc).isoformat(), 'pdf': {'service': service('pdf-translate-reader.service'), 'files': {p: sha(Path(p).read_bytes()) for p in pdf_files}, 'https': https('pdf.fenglin.pro')}, 'hsd': {'service': service('hsd-general-model-paired-review.service'), 'release': str(Path('/opt/hsd-general-model-paired-review/current').resolve()), 'old_session': record('/var/lib/hsd-general-model-paired-review/session.json'), 'evidence_session': record('/var/lib/hsd-general-model-paired-review/evidence-applicability-v1/session.json')}} , ensure_ascii=False, indent=2))
