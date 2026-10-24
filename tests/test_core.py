import unittest
from quietqueue.core import triage, to_markdown

class CoreTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(triage([])['processed'], 0)

    def test_duplicate_suggestions_not_actions(self):
        data = [{'id':'a','title':'Reminder','body':'Please reply with your availability'},
                {'id':'b','title':'Reminder','body':'Please reply with your availability'}]
        res = triage(data)
        self.assertEqual(res['processed'], 2)
        self.assertEqual(len(res['duplicate_pairs']), 1)
        self.assertTrue(all(x['requires_human_approval'] for x in res['items']))
        self.assertTrue('No emails sent' in res['safety'])

    def test_due_date_priority_and_markdown(self):
        res = triage([{'id':'1','title':'Newsletter','body':'Digest for the month'},
                      {'id':'2','title':'Due soon','body':'Submit by 2026-10-25'}])
        self.assertEqual(res['items'][0]['id'], '2')
        self.assertEqual(res['items'][0]['priority'], 'High')
        self.assertIn('| Priority |', to_markdown(res))

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            triage([1])
        with self.assertRaises(ValueError):
            triage([{'title':'hello'}] * 201)

    def test_arabic_no_crash(self):
        res = triage([{'title':'مهمة', 'body':'نرجو الرد على الإيميل'}])
        self.assertEqual(res['processed'], 1)

    def test_invalid_date_ignored(self):
        res = triage([{'title':'hello', 'body':'date 2026-99-99'}])
        self.assertIsNone(res['items'][0]['due_date'])

if __name__ == '__main__': unittest.main()
