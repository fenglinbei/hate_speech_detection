import copy
import unittest

import numpy as np
from sklearn.metrics import f1_score

from diagnostics import general_model_nolabel_analysis as analysis
from diagnostics.general_model_numeric_analysis import candidate_scores
from tests.test_general_model_numeric_coverage_analysis import candidates


class BootstrapTests(unittest.TestCase):
    def test_matches_direct_resampling_and_sklearn_not_mean_individual_f1(self):
        # Differing class composition makes F1 nonlinear in the sampled queries.
        truth = np.array([1,1,1,0,0,0,0])
        predictions = np.array([[1,0,0,0,0,0,1], [1,1,0,1,0,0,1],
                                [1,1,1,1,0,0,0], [1,0,1,1,1,0,0],
                                [1,1,1,0,0,0,1], [1,1,0,0,1,0,0]])
        stats = np.zeros((7,6,7,3),dtype=int)
        for i in range(7):
            for j in range(6):
                for label in range(2):
                    g,p = truth[i]==label,predictions[j,i]==label
                    stats[i,j,label] = [g and p,not g and p,g and not p]
                for label in range(2,7):
                    stats[i,j,label] = stats[i,j,label % 2]
        actual = analysis.paired_f1_bootstrap(stats,repetitions=137,seed=42)
        ids = np.random.default_rng(42).integers(0,7,size=(137,7))
        for contrast,a,b in [('NoCat_without_D',4,1),('NoCat_with_D',5,3)]:
            differences = [f1_score(truth[ix],predictions[a,ix],labels=[0,1],average='macro',zero_division=0)
                           -f1_score(truth[ix],predictions[b,ix],labels=[0,1],average='macro',zero_division=0) for ix in ids]
            row = next(r for r in actual if r['task']=='hate' and r['contrast']==contrast)
            np.testing.assert_allclose(row['descriptive_ci95'],np.quantile(differences,[.025,.975]),atol=1e-12)

    def test_identical_paired_predictions_have_exact_zero_intervals(self):
        rng = np.random.default_rng(10)
        one = rng.integers(0,2,size=(9,1,7,3))
        stats = np.repeat(one,6,axis=1)
        for r in analysis.paired_f1_bootstrap(stats,repetitions=50):
            self.assertEqual(r['difference'],0)
            self.assertEqual(r['descriptive_ci95'],[0,0])


def fixture():
    frame = [{'query_id':str(i),'lex_hit':i%2==0} for i in range(4)]
    gold = {str(i):{'hate':'hate' if i<2 else 'non-hate','group':[] if i==0 else ['Region']} for i in range(4)}
    blocks = [{'query_id':q['query_id'],'task':t,'condition':c,'candidates':candidates(t,j+i)}
              for i,q in enumerate(frame) for t in analysis.TASKS for j,c in enumerate(analysis.CONDITIONS)]
    return frame,gold,blocks


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame,cls.gold,cls.blocks=fixture()
        cls.result=analysis.analyze_blocks(cls.blocks,frame=cls.frame,gold_by_query=cls.gold,
                                          epsilon=.001,expected_query_count=4,bootstrap_replicates=20)

    def test_exact_matrix_four_primary_ci_and_auxiliaries(self):
        r=self.result
        self.assertEqual(r['block_count'],48)
        self.assertEqual(len(r['primary_classification_differences']),4)
        self.assertTrue(all(x['stratum']=='all' for x in r['primary_classification_differences']))
        self.assertTrue(all('descriptive_ci95' not in x for x in r['contrast_summaries']))
        self.assertTrue(any(x['metric']=='answer_sum/gold/nll' for x in r['condition_summaries']))

    def test_missing_duplicate_and_extra_cells_rejected(self):
        for blocks in [self.blocks[:-1],self.blocks+[self.blocks[0]],
                       [{**self.blocks[0],'condition':'unknown'}]+self.blocks[1:]]:
            with self.assertRaises(ValueError):
                analysis.analyze_blocks(blocks,frame=self.frame,gold_by_query=self.gold,
                    epsilon=.001,expected_query_count=4,bootstrap_replicates=2)

    def test_tie_break_uses_ordinal_and_reports_near_tie(self):
        blocks=copy.deepcopy(self.blocks)
        target=next(b for b in blocks if b['query_id']=='0' and b['task']=='hate' and b['condition']=='C0')
        for candidate in target['candidates']:
            n=len(candidate['token_logprobs'])
            candidate['token_logprobs']=[-3./n]*n
            candidate['scores']=candidate_scores(candidate['token_logprobs'],-.25)
        r=analysis.analyze_blocks(blocks,frame=self.frame,gold_by_query=self.gold,
                                 epsilon=.001,expected_query_count=4,bootstrap_replicates=2)
        selected=r['per_query'][0]['conditions']['C0']['hate']['prediction']
        self.assertEqual(selected['ordinal'],0)
        self.assertEqual(selected['tied_top_count'],2)
        self.assertTrue(selected['within_two_epsilon'])


if __name__=='__main__':
    unittest.main()
