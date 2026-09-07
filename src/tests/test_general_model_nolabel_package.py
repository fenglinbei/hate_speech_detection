import copy
import unittest

from diagnostics.general_model_contexts import _render_lexicon
from diagnostics.general_model_nolabel_package import category_removal
from diagnostics.general_model_package import PackageError


class CategoryRemovalTests(unittest.TestCase):
    def setUp(self):
        self.hits = [{"lexicon_id":"id-1","term":"词条","senses":[
            {"sense_id":"s-1","categories":["Region"],
             "definition":"保留定义\n类别：这是定义正文，不能按行删除"},
            {"sense_id":"s-2","categories":["others"],"definition":"第二义项"}]}]
        self.full, _ = _render_lexicon(self.hits,"Full")
        self.tail = '\n\n参考示例：\n输出：["Region"]\n类别：示例原文\n\n待判断文本（JSON 字符串）：\n"类别：query"'
        self.messages = [{"role":"system","content":"系统类别规则保持"},
                         {"role":"user","content":self.full+self.tail}]

    def test_removes_only_schema_fields_preserving_definition_demo_and_query_text(self):
        original = copy.deepcopy(self.messages)
        result,trace,dictionary = category_removal(self.messages,self.hits)
        self.assertEqual(result[0],self.messages[0])
        self.assertEqual(result[1]["content"],dictionary+self.tail)
        self.assertNotIn('类别：["Region"]',dictionary)
        self.assertNotIn('类别：["others"]',dictionary)
        self.assertIn('类别：这是定义正文，不能按行删除',dictionary)
        self.assertIn('第二义项',dictionary)
        self.assertTrue(all(r['definition_visible'] and r['rendered_categories'] is None for r in trace))
        self.assertEqual(self.messages,original)

    def test_rejects_changed_prefix_instead_of_replacing_similar_text_elsewhere(self):
        altered = copy.deepcopy(self.messages)
        altered[1]['content'] = altered[1]['content'].replace('第二义项','改过的义项')
        with self.assertRaises(PackageError):
            category_removal(altered,self.hits)

    def test_empty_dictionary_is_exact_identity(self):
        messages = [{'role':'system','content':'system'},{'role':'user','content':'参考示例：\n类别：保持'}]
        result,trace,text = category_removal(messages,[])
        self.assertEqual(result,messages)
        self.assertEqual(trace,[])
        self.assertEqual(text,'')

    def test_empty_hits_cannot_hide_existing_dictionary(self):
        with self.assertRaises(PackageError):
            category_removal(self.messages,[])


if __name__ == '__main__':
    unittest.main()
