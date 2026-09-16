#!/usr/bin/env python3
"""Deploy the bounded final-annotation reader and commit one explicitly accepted batch.

Run on the existing HSD host. Both sessions are backed up exactly; only the
evidence session is updated. A committed result is never restored from backup.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import tempfile

import update_static_release as common

CHANGED = {
    "tools/general_model_paired_review_ui/evidence.js",
    "tools/general_model_paired_review_ui/evidence_schema.py",
    "tools/general_model_paired_review_ui/evidence_store.py",
    "tools/general_model_paired_review_ui/evidence_finalization.py",
}
WORKER = r'''
import json, sys
from pathlib import Path
release, artifact_path, expected_sha, commit, backup = sys.argv[1:]
release, artifact_path = Path(release), Path(artifact_path)
sys.path[:0] = [str(release), str(release / 'src')]
from build_lex.annotated_lexicon_repair import _session_lock, read_json, file_sha256, write_json
from tools.general_model_paired_review_ui.evidence_store import EvidenceReviewStore
from tools.general_model_paired_review_ui.evidence_finalization import prepare_finalized_session
from tools.general_model_paired_review_ui.evidence_policy import _backup_exact
artifact = read_json(artifact_path)
store = EvidenceReviewStore(bundle_path=release/'evidence/evidence_bundle.json', policy_path=release/'evidence/evidence_policy.json', session_path=Path('/var/lib/hsd-general-model-paired-review/evidence-applicability-v1/session.json'), reviewer_id=artifact['reviewer_id'])
with _session_lock(store.session_path):
    before = store._read()
    assert file_sha256(store.session_path) == expected_sha == artifact['parent_session']['sha256'], 'Latest authoritative bytes differ'
    prepared = prepare_finalized_session(store, before, artifact, file_sha256(artifact_path))
    if commit == 'yes':
        _backup_exact(Path(backup), store.session_path.read_bytes())
        write_json(store.session_path, prepared)
    print(json.dumps({'committed': commit == 'yes', 'revision': prepared['revision'], 'status': store._bootstrap(prepared)['status']}))
'''


def worker(release, artifact, expected, commit=False, backup="unused"):
    result = subprocess.run(["runuser", "-u", "hsd-review", "--", "/usr/bin/python3", "-B", "-c", WORKER,
                             str(release), str(artifact), expected, "yes" if commit else "no", str(backup)],
                            check=True, capture_output=True, text=True, timeout=55)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--expected-session-sha256", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    raw = args.artifact.read_bytes()
    common.require(common.sha(raw) == args.artifact_sha256, "Final artifact hash differs")
    artifact = json.loads(raw)
    common.require(artifact["authorization_text"].strip() and artifact["acceptance_mode"] == "explicit_user_bulk_acceptance", "Explicit final acceptance missing")
    common.require(artifact["parent_session"]["sha256"] == args.expected_session_sha256, "Parent session differs")
    payload = common.archive_payload(args.archive, args.sha256)
    old_release = common.current_release()
    old_payload = common.disk_payload(old_release)
    changed = {p for p in set(payload) | set(old_payload) if payload.get(p) != old_payload.get(p)}
    common.require(changed == CHANGED | {"release_manifest.json"}, "Unrelated release files changed: " + str(sorted(changed)))
    before_check = common.session_snapshot()
    common.require(common.sha(before_check["evidence"]["raw"]) == args.expected_session_sha256, "Concurrent evidence update")
    if args.check_only:
        print(json.dumps({"status": "checked", "changed_files": sorted(CHANGED), "sessions": common.summary(before_check)}))
        return
    unit = common.UNIT.read_bytes()
    release = common.ROOT / "releases" / args.sha256
    with (common.STATE / ".static-release.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        common.require(common.current_release() == old_release, "Release changed concurrently")
        common.install_release(release, payload)
        result_dir = common.STATE / "evidence-applicability-v1/final-results" / args.artifact_sha256
        result_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        account = pwd.getpwnam("hsd-review")
        for path in (result_dir.parent, result_dir):
            os.chown(path, account.pw_uid, account.pw_gid)
            path.chmod(0o700)
        final_path = result_dir / "final-result.json"
        common.private_write(final_path, raw)
        os.chown(final_path, account.pw_uid, account.pw_gid)
        preflight = worker(release, final_path, args.expected_session_sha256)
        common.require(preflight["status"]["confirmed_object_count"] == 1072, "Incomplete final result")
        stopped = False
        backup = None
        try:
            stopped = True
            common.run("systemctl", "stop", common.SERVICE)
            before = common.session_snapshot()
            common.require(common.sha(before["evidence"]["raw"]) == args.expected_session_sha256, "Concurrent update before stop")
            backup = Path(tempfile.mkdtemp(prefix="final-annotation-backup-", dir=common.STATE))
            backup.chmod(0o700)
            for name, entry in before.items():
                common.private_write(backup / (name + "-session.json"), entry["raw"])
                common.require((backup / (name + "-session.json")).read_bytes() == entry["raw"], "Backup bytes differ")
            common.private_write(backup / "metadata.json", common.canonical({"previous_release": str(old_release), "next_release": str(release), "before": common.summary(before), "artifact_sha256": args.artifact_sha256}))
            receipt = worker(release, final_path, args.expected_session_sha256, True, result_dir / "before-session.json")
            common.point_release(release)
            common.run("systemctl", "start", common.SERVICE)
            after = common.wait_ready(before)
            common.require(common.UNIT.read_bytes() == unit, "Unit unexpectedly changed")
            common.require(before["paired"]["raw"] == after["paired"]["raw"], "Paired session unexpectedly changed")
            common.require(after["evidence"]["value"]["revision"] == receipt["revision"], "Review changed during verification; preserve latest state")
            report = {"status": "active_and_finalized", "release": str(release), "backup": str(backup),
                      "final_artifact": str(final_path), "final_artifact_sha256": args.artifact_sha256,
                      "before": common.summary(before), "after": common.summary(after),
                      "new_confirmations": len(artifact["records"]), "native_status": receipt["status"],
                      "case_assessment_confirmations_added": 0, "records_restored_from_backup": False,
                      "nginx_operations": 0, "pdf_service_operations": 0}
            common.private_write(result_dir / "writeback-receipt.json", common.canonical(report))
            os.chown(result_dir / "writeback-receipt.json", account.pw_uid, account.pw_gid)
            print(json.dumps(report))
        except BaseException:
            if stopped:
                common.run("systemctl", "stop", common.SERVICE)
                latest = common.session_snapshot()
                committed = args.artifact_sha256 in latest["evidence"]["value"].get("finalizations", {})
                common.point_release(release if committed else old_release)
                common.run("systemctl", "start", common.SERVICE)
                print(json.dumps({"recovery": "keep_compatible_final_reader" if committed else "code_rolled_back", "records_restored_from_backup": False, "backup": str(backup)}), file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
