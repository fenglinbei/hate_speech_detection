"""Reference eligibility must not infer human reconciliation or treat [] as missing."""
import unittest
from scripts.review.freeze_evidence_analysis import reference_gate

class ReferenceGateTests(unittest.TestCase):
    def test_empty_group_is_valid(self):
        self.assertEqual(reference_gate({'reference_eligible':True},[],[]),(True,[]))
    def test_unresolved_material_needs_explicit_precedence(self):
        ok, reasons=reference_gate({'reference_eligible':True},'non-hate',None)
        self.assertFalse(ok)
        self.assertIn('case_material_difference_pending_confirmation',reasons)
    def test_explicit_precedence_preserves_case_value(self):
        self.assertEqual(reference_gate({'reference_eligible':True},[],['others'],True),(True,[]))
    def test_reconciliation_does_not_bypass_stale_native_gate(self):
        ok,reasons=reference_gate({'reference_eligible':False},[],['others'],True)
        self.assertFalse(ok)
        self.assertEqual(reasons,['native_reference_unavailable'])
    def test_equal_nulls_do_not_make_reference(self):
        self.assertFalse(reference_gate({'reference_eligible':False},None,None)[0])

if __name__=='__main__':unittest.main()
