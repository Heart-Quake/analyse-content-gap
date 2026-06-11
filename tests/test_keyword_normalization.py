import unittest

import pandas as pd

from app import (
    build_keyword_equivalence_map,
    deduplicate_keywords_by_normalized_key,
    enrich_data,
    normalize_keyword_for_grouping,
)


class KeywordNormalizationTest(unittest.TestCase):
    def assert_same_key(self, first_keyword, second_keyword, equivalences=None):
        self.assertEqual(
            normalize_keyword_for_grouping(first_keyword, equivalences),
            normalize_keyword_for_grouping(second_keyword, equivalences),
        )

    def test_accents_are_normalized(self):
        self.assert_same_key('hotel bordeaux', 'hôtel bordeaux')

    def test_regular_singular_plural_variants_are_grouped(self):
        self.assert_same_key('chaussure running', 'chaussures running')
        self.assert_same_key('recette crêpe', 'recettes crêpes')

    def test_simple_gender_variants_are_grouped(self):
        self.assert_same_key('coach sportif', 'coach sportive')
        self.assert_same_key('avocat fiscal', 'avocate fiscale')

    def test_elisions_and_stop_words_are_removed(self):
        self.assert_same_key('assurance habitation', "l’assurance habitation")
        self.assert_same_key('formation seo', 'formation en seo')

    def test_hyphens_are_normalized(self):
        self.assert_same_key('après-vente', 'apres vente')
        self.assert_same_key('prêt-à-porter', 'pret a porter')

    def test_word_order_is_neutralized(self):
        self.assert_same_key('agence seo paris', 'seo paris agence')
        self.assert_same_key('formation excel debutant', 'excel debutant formation')

    def test_custom_equivalences_are_applied(self):
        equivalences = build_keyword_equivalence_map('rh=ressources humaines')
        self.assert_same_key('rh', 'ressources humaines', equivalences)

    def test_highest_volume_keyword_is_kept(self):
        df = pd.DataFrame([
            {
                'Keyword': 'chaussures running',
                'Volume': 100,
                'KD': 10,
                'CPC': 0.2,
                'SERP features': '',
                'client.com: Position': 4,
            },
            {
                'Keyword': 'chaussure running',
                'Volume': 350,
                'KD': 60,
                'CPC': 0.4,
                'SERP features': '',
                'client.com: Position': 12,
            },
            {
                'Keyword': 'agence seo paris',
                'Volume': 90,
                'KD': 20,
                'CPC': 1.1,
                'SERP features': '',
                'client.com: Position': 8,
            },
        ])

        result = deduplicate_keywords_by_normalized_key(df, 'client.com')

        self.assertEqual(len(result), 2)
        running_row = result[result['Keyword normalisé'] == normalize_keyword_for_grouping('chaussure running')].iloc[0]
        self.assertEqual(running_row['Keyword'], 'chaussure running')
        self.assertEqual(running_row['Nb variantes regroupées'], 2)
        self.assertIn('chaussures running', running_row['Variantes regroupées'])
        self.assertIn('chaussure running', running_row['Variantes regroupées'])

    def test_ties_are_resolved_by_kd_then_client_position(self):
        df = pd.DataFrame([
            {
                'Keyword': 'hotel paris',
                'Volume': 200,
                'KD': 30,
                'CPC': 1.2,
                'SERP features': '',
                'client.com: Position': 2,
            },
            {
                'Keyword': 'hôtels paris',
                'Volume': 200,
                'KD': 10,
                'CPC': 1.1,
                'SERP features': '',
                'client.com: Position': 9,
            },
        ])

        result = deduplicate_keywords_by_normalized_key(df, 'client.com')

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['Keyword'], 'hôtels paris')

    def test_keyword_audit_columns_are_moved_to_the_end_after_enrichment(self):
        df = pd.DataFrame([
            {
                'Keyword': 'chaussures running',
                'Keyword normalisé': normalize_keyword_for_grouping('chaussure running'),
                'Nb variantes regroupées': 2,
                'Variantes regroupées': 'chaussures running | chaussure running',
                'Volume': 100,
                'KD': 10,
                'CPC': 0.2,
                'SERP features': '',
                'client.com: Position': 4,
                'client.com: URL': 'https://client.com/running',
                'client.com: Traffic': 20,
            }
        ])

        enriched_df = enrich_data(df, 'client.com', custom_brands=[])

        self.assertEqual(
            enriched_df.columns.tolist()[-3:],
            ['Keyword normalisé', 'Nb variantes regroupées', 'Variantes regroupées']
        )


if __name__ == '__main__':
    unittest.main()
