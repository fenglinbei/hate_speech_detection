"""Join disjoint successful checks; preserve the timeout and both source records."""
from pathlib import Path
import hashlib
import json
import shutil

WORK = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


original = WORK / "browser-live-01/failure.json"
continuation = WORK / "browser-continuation-live-02/browser.json"
first = read(original)
second = read(continuation)
local = read(WORK / "browser-local-01/browser.json")
local_resume = read(WORK / "browser-continuation-local-02/browser.json")
pins = read(WORK / "browser-continuation-v2-source-pins.json")
assert sha(original) == pins["preserved_failure_sha256"]
assert sha(pins["file"]["path"]) == pins["file"]["sha256"]
assert sha(pins["original_script"]) == pins["original_script_sha256"]
assert "Timeout 180000ms exceeded" in first["error"]
assert first["errors"] == []
assert second["status"] == local["status"] == local_resume["status"] == "pass"
assert second["errors"] == []
checks = first["checks"] + second["checks"]


def scientific_fields(items):
    return [{key: value for key, value in item.items() if key != "cache"} for item in items]


assert scientific_fields(checks) == scientific_fields(local["checks"])
assert scientific_fields(second["checks"]) == scientific_fields(local_resume["checks"])
assert sum(item["cells"] for item in first["checks"]) == 10648
assert second["checked_values"] == 2394
assert sum(item["cells"] for item in checks) == local["checked_values"] == 13042
out = WORK / "browser-live-complete-01"
out.mkdir()
for name in ["mean.svg", "layers.svg", "desktop-mean.png", "mobile.png"]:
    shutil.copyfile(original.parent / name, out / name)
for name in ["hehe-mean.svg", "hehe-desktop-mean.png", "hehe-mobile.png"]:
    shutil.copyfile(continuation.parent / name, out / name)
receipt = dict(second)
receipt.update({
    "status": "pass",
    "scope": "Complete live HTTPS checklist, joined from disjoint successful segments",
    "checked_values": 13042,
    "checks": checks,
    "full_local_checklist_exactly_matched": True,
    "binary_requests": [p for p in first["payloads"] if p.endswith((".f32", ".f64"))] + second["binary_requests"],
    "preserved_prior_attempt_error": first["error"],
    "evidence": [
        {"path": str(original), "sha256": sha(original), "successful_values": 10648},
        {"path": str(continuation), "sha256": sha(continuation), "successful_values": 2394},
    ],
    "timings_note": "Only continuation timing was recorded; the first segment ended at a network timeout.",
})
with (out / "browser.json").open("x", encoding="utf-8") as f:
    f.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": "pass", "checked_values": 13042,
                  "checklist_matches_full_local_run": True}))
