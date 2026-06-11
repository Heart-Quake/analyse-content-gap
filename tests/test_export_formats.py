import json
import unittest

import pandas as pd

from app import (
    dataframe_to_json_string,
    format_integer_display,
    prepare_results_dataframe_for_display,
)


class ExportFormatsTest(unittest.TestCase):
    def test_dataframe_to_json_string_outputs_valid_records(self):
        df = pd.DataFrame([
            {
                'Keyword': 'hôtel paris',
                'Volume': 120,
                'URL': None,
            },
            {
                'Keyword': 'автоплюс',
                'Volume': None,
                'URL': 'https://example.com',
            },
        ])

        json_output = dataframe_to_json_string(df)
        payload = json.loads(json_output)

        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]['Keyword'], 'hôtel paris')
        self.assertIsNone(payload[0]['URL'])
        self.assertEqual(payload[1]['Keyword'], 'автоплюс')
        self.assertIsNone(payload[1]['Volume'])

    def test_format_integer_display_removes_useless_decimals(self):
        self.assertEqual(format_integer_display(700.000000), '700')
        self.assertEqual(format_integer_display(12500.4), '12 500')
        self.assertEqual(format_integer_display(None), '')

    def test_prepare_results_dataframe_for_display_formats_volume_without_decimals(self):
        df = pd.DataFrame([
            {'Keyword': 'gamme cupra', 'Volume': 2700.000000},
            {'Keyword': 'actualité renault', 'Volume': 600.000000},
        ])

        display_df = prepare_results_dataframe_for_display(df)

        self.assertEqual(display_df['Volume'].tolist(), ['2 700', '600'])
        self.assertTrue(pd.api.types.is_object_dtype(display_df['Volume']))
        self.assertTrue(pd.api.types.is_float_dtype(df['Volume']))

    def test_prepare_results_dataframe_for_display_normalizes_position_columns(self):
        df = pd.DataFrame([
            {'Keyword': 'gamme cupra', 'client.fr: Position': 4},
            {'Keyword': 'actualité renault', 'client.fr: Position': ''},
        ])

        display_df = prepare_results_dataframe_for_display(df)

        self.assertEqual(display_df['client.fr: Position'].tolist(), ['4', ''])


if __name__ == '__main__':
    unittest.main()
