"""Remove only this verified release's temporary credential, preview and upload."""
from pathlib import Path
import hashlib
import importlib.util
import json
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
WORK = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


assert read(WORK / "browser-live-complete-01/browser.json")["status"] == "pass"
assert read(WORK / "external-https.json")["status"] == "pass"
assert read(WORK / "activation-receipt.json")["status"] == "deployed"
assert read(WORK / "remote-receipts/before.json") == read(WORK / "remote-receipts/after.json")

credential = WORK / "runtime/credential.json"
assert credential.is_file() and not credential.is_symlink()
credential.unlink()
assert not credential.exists()
save(WORK / "credential-cleanup.json", {
    "temporary_plaintext_removed": True,
    "path": str(credential),
    "at_unix": time.time(),
})

spec = importlib.util.spec_from_file_location(
    "owned_process_helpers", ROOT / "scripts/review/run_hehe_sense_context_window_v1.py"
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
owner = read(WORK / "runtime/preview-owner.json")
before = helpers.identity(owner["pid"])
if before is not None:
    assert helpers.same_process(owner), "Preview identity mismatch; no signal sent."
    assert helpers.send_owned(owner, signal.SIGTERM)
    for _ in range(100):
        if helpers.identity(owner["pid"]) is None:
            break
        time.sleep(0.1)
assert helpers.identity(owner["pid"]) is None
save(WORK / "preview-cleanup.json", {
    "owned_preview_absent": True,
    "owner": owner,
    "signal": "SIGTERM" if before else None,
    "at_unix": time.time(),
})

remote = r'''
from pathlib import Path
import fcntl,hashlib,json,shutil,time
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4194304),b''):h.update(chunk)
    return h.hexdigest()
base=Path('/opt/hsd-case-attention')
release=base/'releases/incremental-20260919-02'
receipt=base/'deployments/incremental-20260919-02/receipt.json'
archive=Path('/tmp/hsd-hehe-incremental-20260919-02.tar')
with Path('/var/lib/hsd-general-model-paired-review/.static-release.lock').open('a+b') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads(receipt.read_text())['status']=='deployed'
    assert sha('/etc/nginx/sites-available/hsd.fenglin.pro')=='09f4b0e53933031752badeca3774c5c7a166fb03e97009320b4260f57f60af40'
    assert sha(release/'release-manifest.json')=='f441b364ec767a5aabf8ea2924b50bf84656891c2b0e9e655b664a0dcc679976'
    assert archive.is_file() and not archive.is_symlink()
    size=archive.stat().st_size;digest=sha(archive)
    assert size==191016960
    assert digest=='ef4ecaf5cfe0c9df77afb7b531a067a7f6fd130ceb499eabadbf99510e1678b0'
    free_before=shutil.disk_usage(base).free
    archive.unlink()
    assert not archive.exists()
    print(json.dumps({'archive_removed':True,'removed_only_own_uploaded_archive':str(archive),
        'bytes':size,'sha256':digest,'free_before':free_before,'disk_free':shutil.disk_usage(base).free,
        'local_archived_copy_retained':True,'old_releases_retained':True,'at_unix':time.time()}))
'''
result = subprocess.run(
    ["ssh", "digitalocean-sgp", "python3 -"], input=remote, text=True,
    capture_output=True, check=True, timeout=60,
)
cleanup = json.loads(result.stdout)
assert cleanup["archive_removed"]
local_archive = WORK / "prepared/delta.tar"
assert local_archive.is_file() and local_archive.stat().st_size == cleanup["bytes"]
save(WORK / "archive-cleanup.json", cleanup)
print(json.dumps({"status": "cleaned", "credential_absent": True,
                  "preview_absent": True, "archive_removed": True,
                  "server_free_bytes": cleanup["disk_free"]}))
