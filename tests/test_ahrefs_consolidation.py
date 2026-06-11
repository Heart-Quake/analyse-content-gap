from io import BytesIO
import unittest

import pandas as pd

from app import (
    build_ahrefs_url_consolidation,
    detect_ahrefs_consolidation_columns,
    process_ahrefs_consolidation_file,
)


class NamedBytesIO(BytesIO):
    def __init__(self, content: bytes, name: str):
        super().__init__(content)
        self.name = name


class AhrefsConsolidationTest(unittest.TestCase):
    def test_detects_classic_organic_keywords_columns(self):
        df = pd.DataFrame(columns=['Keyword', 'Volume', 'Current URL', 'Current position'])

        detected = detect_ahrefs_consolidation_columns(df)

        self.assertEqual(detected['format_type'], 'organic_keywords')
        self.assertEqual(detected['url'], 'Current URL')
        self.assertEqual(detected['position'], 'Current position')

    def test_detects_content_gap_domain_columns(self):
        df = pd.DataFrame(columns=[
            'Keyword',
            'Volume',
            'client.com: URL',
            'client.com: Organic Position',
            'competitor.com: URL',
            'competitor.com: Organic Position',
        ])

        detected = detect_ahrefs_consolidation_columns(df, preferred_domain='client.com')

        self.assertEqual(detected['format_type'], 'content_gap')
        self.assertEqual(detected['url'], 'client.com: URL')
        self.assertEqual(detected['position'], 'client.com: Organic Position')

    def test_processes_tsv_export_and_consolidates_by_url(self):
        content = (
            'Keyword\tVolume\tCurrent URL\tCurrent position\n'
            'chaussure running\t1000\thttps://example.com/running\t3\n'
            'basket running\t500\thttps://example.com/running\t6\n'
            'chaussure trail\t300\thttps://example.com/trail\t4\n'
        ).encode('utf-8')
        uploaded_file = NamedBytesIO(content, 'example.com-organic-keywords.csv')

        cleaned_df = process_ahrefs_consolidation_file(uploaded_file)
        result_df = build_ahrefs_url_consolidation(cleaned_df)

        self.assertEqual(len(result_df), 2)
        running_row = result_df[result_df['URL'] == 'https://example.com/running'].iloc[0]
        self.assertEqual(running_row['Top mot-clé'], 'chaussure running')
        self.assertEqual(running_row['Volume du top mot-clé'], 1000)
        self.assertEqual(running_row['Volume total'], 1500)
        self.assertEqual(running_row['Nb mots-clés'], 2)
        self.assertEqual(running_row['Position moyenne'], 4.5)
        self.assertIn('basket running', running_row['Mots-clés'])

    def test_empty_urls_are_removed_before_consolidation(self):
        df = pd.DataFrame([
            {'keyword': 'kw 1', 'volume': 100, 'position': 2, 'url': 'https://example.com/a'},
            {'keyword': 'kw 2', 'volume': 200, 'position': 4, 'url': ''},
        ])

        result_df = build_ahrefs_url_consolidation(df[df['url'] != ''])

        self.assertEqual(len(result_df), 1)
        self.assertEqual(result_df.iloc[0]['URL'], 'https://example.com/a')


if __name__ == '__main__':
    unittest.main()
