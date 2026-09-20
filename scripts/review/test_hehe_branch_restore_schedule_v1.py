#!/usr/bin/env python3
"""CPU checks for one-shot time gate, cancellation, pipeline order and failures."""
import argparse,importlib.util,json,sys,tempfile,unittest
from contextlib import ExitStack
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('restore_schedule',ROOT/'reviews/hehe-branch-restore-v1/schedule-01/schedule.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


class Checks(unittest.TestCase):
    def test_time_gate_and_cancel(self):
        current=[100.];sleeps=[]
        def sleep(seconds):sleeps.append(seconds);current[0]+=seconds
        self.assertTrue(s.wait_until(195.,lambda:False,lambda:current[0],sleep))
        self.assertEqual(current[0],195.);self.assertEqual(sleeps,[30.,30.,30.,5.])
        self.assertFalse(s.wait_until(250.,lambda:True,lambda:current[0],sleep))
        self.assertEqual(current[0],195.)

    def test_pipeline_order_failure_and_stop(self):
        names=[name for name,_ in s.pipeline_commands()]
        self.assertEqual(names,['GPU','analysis','independent_audit','readable_report','closeout'])
        calls=[]
        def stage(name,cmd):
            calls.append(name)
            if name=='independent_audit':raise RuntimeError('fixture audit failure')
        with self.assertRaises(RuntimeError):s.run_pipeline(stage,lambda:False)
        self.assertEqual(calls,names[:3])
        calls=[]
        self.assertFalse(s.run_pipeline(lambda name,cmd:calls.append(name),lambda:len(calls)==1))
        self.assertEqual(calls,['GPU'])

    def test_actual_entrypoint_once_and_closed_wait(self):
        with tempfile.TemporaryDirectory(prefix='hehe-restore-schedule-test-') as td,ExitStack() as stack:
            here=Path(td);launch=here/'launch';launch.mkdir();public=here/'public';public.mkdir()
            for name,value in [('HERE',here),('LAUNCH',launch),('RUN',here/'run'),('stopping',False),('claimed',False)]:
                stack.enter_context(patch.object(s,name,value))
            stack.enter_context(patch.object(s.c,'PUBLIC',public))
            stack.enter_context(patch.object(s.c,'validate',return_value=None))
            stack.enter_context(patch.object(s.signal,'signal'))
            now=datetime.now(timezone.utc).isoformat()
            s.c.write(here/'config.json',{'not_before':now});s.c.write(here/'source-pins.json',{'files':[]})
            calls=[]
            def step(name,cmd):
                calls.append(name)
                if name=='closeout':s.c.write(public/'results-current.json',{'report':'synthetic-only','closeout_manifest':{'synthetic':True}})
            stack.enter_context(patch.object(s,'stage',side_effect=step))
            s.main()
            self.assertEqual(calls,[name for name,_ in s.pipeline_commands()])
            self.assertEqual(s.c.read(here/'state.json')['status'],'complete')
            original=(here/'state.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'already started'):s.main()
            self.assertEqual((here/'state.json').read_bytes(),original)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt={'status':'pass' if result.wasSuccessful() else 'fail','tests':result.testsRun,
        'real_GPU_or_scheduler_started':False,'failures':[str(e) for _,e in result.failures+result.errors],
        'implementation_snapshot':[s.c.info(Path(__file__)),s.c.info(Path(s.__file__))]}
    s.c.write(a.output,receipt);print(json.dumps(receipt));sys.exit(0 if result.wasSuccessful() else 1)
