#!/usr/bin/env python3
"""Regression: authored report may differ; every copied scientific byte must match."""
import argparse,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import report_hehe_branch_restore_v2 as r


class Checks(unittest.TestCase):
    def test_editorial_difference_and_copied_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'results';b=Path(td)/'report';a.mkdir();b.mkdir()
            for name in ['results.json','all-trajectories.tsv','figure.svg']:
                (a/name).write_text(name);(b/name).write_text(name)
            (a/'REPORT.md').write_text('original');(b/'REPORT.md').write_text('new interpretation')
            names=r.copied_names(a);self.assertEqual(len(names),3)
            r.verify_copied_assets(a,b,names)
            (b/'results.json').write_text('corrupted')
            with self.assertRaisesRegex(ValueError,'copied bytes changed'):r.verify_copied_assets(a,b,names)

    def test_missing_copy_is_not_waived(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'results';b=Path(td)/'report';a.mkdir();b.mkdir()
            (a/'results.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'inventory changed'):r.verify_copied_assets(a,b,[])
            with self.assertRaises(FileNotFoundError):r.verify_copied_assets(a,b,['results.json'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt={'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,'GPU_touched':False,
        'source':r.c.info(Path(r.__file__)),'test_source':r.c.info(Path(__file__)),
        'failures':[str(e) for _,e in result.failures+result.errors]}
    r.c.write(a.output,receipt);print(json.dumps(receipt));sys.exit(0 if result.wasSuccessful() else 1)
