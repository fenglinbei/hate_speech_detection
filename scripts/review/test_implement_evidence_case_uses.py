"""Verify intervention boundaries with synthetic material only."""
from copy import deepcopy
import unittest

from scripts.review.implement_evidence_case_uses import source_partition, surviving_entries, messages_for


class SourcePartitionTests(unittest.TestCase):
    def setUp(self):
        self.graph = {'demo_order':['a','b','c'], 'entry_order':['query','shared','unique'],
                      'query_entry_ids':['query'],
                      'demo_entry_ids':{'a':['query','shared','unique'],'b':['shared'],'c':[]},
                      'hit_edges':[
                          {'record_id':'q','entry_id':'query'},
                          {'record_id':'a-q','entry_id':'query'},
                          {'record_id':'a-shared','entry_id':'shared'},
                          {'record_id':'b-shared','entry_id':'shared'},
                          {'record_id':'a-unique','entry_id':'unique'}]}

    def test_only_unique_source_contribution_can_be_removed(self):
        result=source_partition(self.graph,['a'])
        self.assertEqual(result['retained_demo_ids'],['b','c'])
        self.assertEqual(result['L_R'],['query','shared'])
        self.assertEqual(result['L_U'],['unique'])
        self.assertEqual(result['shared_selected_entry_ids'],['query','shared'])

    def test_empty_lexicon_factor_is_explicit(self):
        result=source_partition(self.graph,['c'])
        self.assertEqual(result['L_U'],[])
        self.assertFalse(result['dictionary_factor_effective'])

    def test_masking_one_shared_edge_keeps_entry(self):
        self.assertEqual(surviving_entries(self.graph,['a-shared']),['query','shared','unique'])
        self.assertEqual(surviving_entries(self.graph,['a-shared','b-shared']),['query','unique'])

    def test_unknown_or_duplicate_selection_rejected(self):
        with self.assertRaises(ValueError):source_partition(self.graph,['a','a'])
        with self.assertRaises(ValueError):source_partition(self.graph,['absent'])
        with self.assertRaises(ValueError):surviving_entries(self.graph,['absent'])


class InputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.entries={'one':{'lexicon_id':'one','term':'词','senses':[{'sense_id':'s1','definition':'固定释义','categories':['others']}]}}
        self.materials={'demo:a':{'source':{'text':'保留文本：输出：不会被替换','original_answer':{'hate':'hate','group':['others']}}},
                        'demo:b':{'source':{'text':'另一段文本','original_answer':{'hate':'non-hate','group':[]}}}}
        self.source={'messages':[{'role':'system','content':'fixed instructions'}, {'role':'user','content':'ORIGINAL_SLOT'}],
                     'prompt_text':'PREFIXORIGINAL_SLOTSUFFIX'}

    def test_demo_removal_keeps_dictionary_query_and_remaining_answer(self):
        arm={'entry_ids':['one'],'demo_ids':['b'],'answer_overrides':{}}
        messages,prompt=messages_for(self.source,'独立查询',self.entries,self.materials,arm,'hate')
        self.assertIn('定义：固定释义',prompt)
        self.assertIn('文本：另一段文本\n输出："non-hate"',prompt)
        self.assertNotIn('不会被替换',prompt)
        self.assertIn('"独立查询"',prompt)
        self.assertEqual(messages[0],self.source['messages'][0])
        self.assertTrue(prompt.startswith('PREFIX') and prompt.endswith('SUFFIX'))

    def test_answer_change_is_task_and_demo_specific(self):
        before={'entry_ids':['one'],'demo_ids':['a','b'],'answer_overrides':{}}
        after=deepcopy(before);after['answer_overrides']={'a':['Racism']}
        old=messages_for(self.source,'查询',self.entries,self.materials,before,'group')[1]
        new=messages_for(self.source,'查询',self.entries,self.materials,after,'group')[1]
        self.assertEqual(new,old.replace('输出：["others"]','输出：["Racism"]'))
        self.assertEqual(self.materials['demo:a']['source']['original_answer']['group'],['others'])

    def test_ambiguous_template_slot_rejected(self):
        self.source['prompt_text']='ORIGINAL_SLOTORIGINAL_SLOT'
        with self.assertRaises(ValueError):
            messages_for(self.source,'查询',self.entries,self.materials,{'entry_ids':[],'demo_ids':[],'answer_overrides':{}},'hate')


if __name__=='__main__':unittest.main()
