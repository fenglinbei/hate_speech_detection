from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_lex.terminology_provider_pricing import (  # noqa: E402
    APPLICABILITY_SCOPE,
    PROVIDER_POLICY,
    RATE_POLICY_ID,
    ProviderPricingError,
    pricing_projection_for_successor_plan,
    validate_provider_pricing_bundle,
    verify_provider_pricing_evidence,
    write_verified_pricing_bundle,
)
from data.training_artifacts import canonical_sha256  # noqa: E402


CAPTURED_AT = "2026-08-30T00:00:00Z"


class ProviderPricingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "schemas").mkdir()
        for name in (
            "wp3_provider_pricing_evidence_v1.schema.json",
            "wp3_provider_pricing_verification_bundle_v1.schema.json",
        ):
            shutil.copyfile(
                REPOSITORY_ROOT / "schemas" / name,
                self.root / "schemas" / name,
            )
        (self.root / "pricing-snapshots").mkdir()
        self.snapshot_texts = {
            "glm_flash": (
                "Official GLM pricing\n"
                "Requested model: glm-5.3-flash\n"
                "Input RMB per million tokens: 3\n"
                "Output RMB per million tokens: 9\n"
            ),
            "deepseek_flash": (
                "Official DeepSeek peak pricing\n"
                "Requested model: deepseek-v4-flash\n"
                "Peak cache-miss input RMB per million tokens: 2\n"
                "Peak output RMB per million tokens: 8\n"
            ),
        }
        self.paths: dict[str, Path] = {}
        for provider_id, text in self.snapshot_texts.items():
            path = self.root / "pricing-snapshots" / f"{provider_id}.txt"
            path.write_bytes(text.encode("utf-8"))
            self.paths[provider_id] = path
        self.evidence = self._evidence()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _receipt(
        self,
        provider_id: str,
        *,
        input_rate: str,
        output_rate: str,
    ) -> dict:
        text = self.snapshot_texts[provider_id]
        policy = PROVIDER_POLICY[provider_id]
        filename = f"pricing-snapshots/{provider_id}.txt"
        if provider_id == "glm_flash":
            url = "https://bigmodel.cn/pricing"
            input_quote = "Input RMB per million tokens: 3"
            output_quote = "Output RMB per million tokens: 9"
        else:
            url = "https://api-docs.deepseek.com/quick_start/pricing"
            input_quote = "Peak cache-miss input RMB per million tokens: 2"
            output_quote = "Peak output RMB per million tokens: 8"
        return {
            "provider_id": provider_id,
            "requested_model": policy["requested_model"],
            "source_url": url,
            "snapshot_path": filename,
            "snapshot_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "snapshot_size_bytes": len(text.encode("utf-8")),
            "captured_at": CAPTURED_AT,
            "rate_policy_application": {
                "applicability_scope": APPLICABILITY_SCOPE,
                "input_basis": policy["input_basis"],
                "output_basis": policy["output_basis"],
            },
            "evidence": {
                "model": {
                    "exact_text": f"Requested model: {policy['requested_model']}",
                    "occurrence_ordinal": 1,
                },
                "input_rate": {
                    "exact_text": input_quote,
                    "occurrence_ordinal": 1,
                    "numeric_text": input_rate,
                    "semantic_label": "input_rmb_per_million",
                },
                "output_rate": {
                    "exact_text": output_quote,
                    "occurrence_ordinal": 1,
                    "numeric_text": output_rate,
                    "semantic_label": "output_rmb_per_million",
                },
            },
        }

    def _evidence(self) -> dict:
        return {
            "schema_version": "wp3-provider-pricing-evidence/v1",
            "scope": "development-only",
            "scientific_eligible": False,
            "sealed": False,
            "currency": "RMB",
            "captured_at": CAPTURED_AT,
            "rate_policy": RATE_POLICY_ID,
            "source_receipts": [
                self._receipt("glm_flash", input_rate="3", output_rate="9"),
                self._receipt("deepseek_flash", input_rate="2", output_rate="8"),
            ],
            "provider_rates": {
                "glm_flash": {
                    "input_rmb_per_million": "3",
                    "output_rmb_per_million": "9",
                },
                "deepseek_flash": {
                    "input_rmb_per_million": "2",
                    "output_rmb_per_million": "8",
                },
            },
        }

    def test_valid_evidence_builds_sanitized_content_bound_projection(self) -> None:
        bundle = verify_provider_pricing_evidence(
            self.evidence, workspace_root=self.root
        )
        self.assertEqual(bundle["scope"], "development-only")
        self.assertIs(bundle["scientific_eligible"], False)
        self.assertIs(bundle["sealed"], False)
        receipt = bundle["verification_receipt"]
        self.assertIs(receipt["network_access_performed"], False)
        self.assertIs(receipt["verified"], True)
        self.assertEqual(
            bundle["verification_receipt_sha256"], canonical_sha256(receipt)
        )
        self.assertEqual(
            receipt["pricing_projection_sha256"],
            canonical_sha256(bundle["pricing_projection"]),
        )
        self.assertEqual(
            [row["provider_id"] for row in bundle["pricing_projection"]["source_receipts"]],
            ["glm_flash", "deepseek_flash"],
        )
        self.assertNotIn(
            "snapshot_path", bundle["pricing_projection"]["source_receipts"][0]
        )
        self.assertEqual(
            pricing_projection_for_successor_plan(
                evidence=self.evidence, workspace_root=self.root
            ),
            bundle["pricing_projection"],
        )

    def test_snapshot_tamper_is_rejected_even_if_length_is_unchanged(self) -> None:
        original = self.paths["glm_flash"].read_bytes()
        self.paths["glm_flash"].write_bytes(original.replace(b"tokens: 3", b"tokens: 4"))
        with self.assertRaisesRegex(ProviderPricingError, "SHA-256"):
            verify_provider_pricing_evidence(
                self.evidence, workspace_root=self.root
            )

    def test_unsafe_and_symlink_snapshot_paths_are_rejected(self) -> None:
        for unsafe in ("../outside.txt", "/tmp/outside.txt", "a//b.txt", "a\\b.txt"):
            with self.subTest(path=unsafe):
                changed = copy.deepcopy(self.evidence)
                changed["source_receipts"][0]["snapshot_path"] = unsafe
                with self.assertRaises(ProviderPricingError):
                    verify_provider_pricing_evidence(changed, workspace_root=self.root)

        target = self.paths["glm_flash"]
        link = self.root / "pricing-snapshots" / "glm-link.txt"
        os.symlink(target.name, link)
        changed = copy.deepcopy(self.evidence)
        changed["source_receipts"][0]["snapshot_path"] = (
            "pricing-snapshots/glm-link.txt"
        )
        with self.assertRaisesRegex(ProviderPricingError, "symlink"):
            verify_provider_pricing_evidence(changed, workspace_root=self.root)

    def test_arbitrary_claimed_hash_does_not_count_as_verified(self) -> None:
        changed = copy.deepcopy(self.evidence)
        changed["source_receipts"][0]["snapshot_sha256"] = "f" * 64
        with self.assertRaisesRegex(ProviderPricingError, "SHA-256"):
            verify_provider_pricing_evidence(changed, workspace_root=self.root)

    def test_exact_evidence_and_selected_rate_must_replay(self) -> None:
        changed = copy.deepcopy(self.evidence)
        changed["source_receipts"][0]["evidence"]["input_rate"][
            "exact_text"
        ] = "Claimed but absent input rate: 3"
        with self.assertRaisesRegex(ProviderPricingError, "absent"):
            verify_provider_pricing_evidence(changed, workspace_root=self.root)

        changed = copy.deepcopy(self.evidence)
        changed["provider_rates"]["glm_flash"]["input_rmb_per_million"] = "4"
        with self.assertRaisesRegex(ProviderPricingError, "does not match"):
            verify_provider_pricing_evidence(changed, workspace_root=self.root)

        changed = copy.deepcopy(self.evidence)
        changed["source_receipts"][0]["evidence"]["model"][
            "exact_text"
        ] = "Official GLM pricing"
        with self.assertRaisesRegex(ProviderPricingError, "requested model"):
            verify_provider_pricing_evidence(changed, workspace_root=self.root)

    def test_only_provider_specific_official_https_hosts_are_accepted(self) -> None:
        cases = (
            "http://bigmodel.cn/pricing",
            "https://www.bigmodel.cn/pricing",
            "https://bigmodel.cn/other",
            "https://bigmodel.cn/pricing?campaign=test",
            "https://docs.bigmodel.cn/cn/guide/models/pricing",
            "https://example.com/pricing",
            "https://api-docs.deepseek.com.evil.invalid/pricing",
        )
        for url in cases:
            with self.subTest(url=url):
                changed = copy.deepcopy(self.evidence)
                changed["source_receipts"][0]["source_url"] = url
                with self.assertRaises(ProviderPricingError):
                    verify_provider_pricing_evidence(changed, workspace_root=self.root)

        changed = copy.deepcopy(self.evidence)
        changed["source_receipts"][0]["source_url"] = (
            "https://api-docs.deepseek.com/quick_start/pricing"
        )
        with self.assertRaises(ProviderPricingError):
            verify_provider_pricing_evidence(changed, workspace_root=self.root)

    def test_glm_projection_preserves_exact_confirmed_product_pricing_page(self) -> None:
        bundle = verify_provider_pricing_evidence(
            self.evidence, workspace_root=self.root
        )
        glm_receipt = bundle["pricing_projection"]["source_receipts"][0]
        self.assertEqual(glm_receipt["provider_id"], "glm_flash")
        self.assertEqual(glm_receipt["source_url"], "https://bigmodel.cn/pricing")

    def test_conservative_policy_and_capture_time_are_frozen(self) -> None:
        changed = copy.deepcopy(self.evidence)
        changed["source_receipts"][1]["rate_policy_application"][
            "input_basis"
        ] = "cache-hit-input/v1"
        with self.assertRaisesRegex(ProviderPricingError, "rate basis"):
            verify_provider_pricing_evidence(changed, workspace_root=self.root)

        changed = copy.deepcopy(self.evidence)
        changed["source_receipts"][1]["captured_at"] = "2026-08-31T00:00:00Z"
        with self.assertRaisesRegex(ProviderPricingError, "capture time"):
            verify_provider_pricing_evidence(changed, workspace_root=self.root)

    def test_bundle_validation_reruns_snapshots_and_rejects_tamper(self) -> None:
        bundle = verify_provider_pricing_evidence(
            self.evidence, workspace_root=self.root
        )
        bundle_path = self.root / "verified-pricing.json"
        write_verified_pricing_bundle(bundle_path, bundle)
        self.assertEqual(
            validate_provider_pricing_bundle(
                bundle_path,
                evidence=self.evidence,
                workspace_root=self.root,
            ),
            bundle,
        )
        changed = json.loads(bundle_path.read_text(encoding="utf-8"))
        changed["pricing_projection"]["provider_rates"]["glm_flash"][
            "input_rmb_per_million"
        ] = "1"
        bundle_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ProviderPricingError, "differs"):
            validate_provider_pricing_bundle(
                bundle_path,
                evidence=self.evidence,
                workspace_root=self.root,
            )

    def test_cli_verify_and_validate_are_offline_and_replaying(self) -> None:
        evidence_path = self.root / "pricing-evidence.json"
        evidence_path.write_text(
            json.dumps(self.evidence, ensure_ascii=False), encoding="utf-8"
        )
        output = self.root / "verified-pricing.json"
        script = REPOSITORY_ROOT / "scripts/stage1/wp3_provider_pricing.py"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(SRC_ROOT)
        verify = subprocess.run(
            [
                sys.executable,
                str(script),
                "--workspace-root",
                str(self.root),
                "verify-pricing",
                "--evidence",
                evidence_path.name,
                "--output",
                output.name,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(verify.returncode, 0, verify.stderr)
        self.assertTrue(output.is_file())
        validate = subprocess.run(
            [
                sys.executable,
                str(script),
                "--workspace-root",
                str(self.root),
                "validate-pricing",
                "--evidence",
                evidence_path.name,
                "--bundle",
                output.name,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(validate.returncode, 0, validate.stderr)
        self.assertIs(json.loads(validate.stdout)["verified"], True)


if __name__ == "__main__":
    unittest.main()
