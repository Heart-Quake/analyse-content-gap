import unittest

import pandas as pd

from app import (
    add_ngram_clusters_to_view,
    build_ngram_visualization_summary,
    create_ngram_cluster_bar_chart,
    extract_keyword_ngrams,
    merge_keyword_equivalence_rules,
    normalize_keyword_for_ngrams,
)


class KeywordNgramClusteringTest(unittest.TestCase):
    def test_ngram_normalization_handles_accents_hyphens_and_stop_words(self):
        tokens = normalize_keyword_for_ngrams("L’après-vente hôtel")
        self.assertEqual(tokens, ['apres', 'vente', 'hotel'])

    def test_extracts_bigrams_and_trigrams_from_normalized_tokens(self):
        tokens = normalize_keyword_for_ngrams('agence seo paris')
        ngrams = extract_keyword_ngrams(tokens)
        observed_labels = [ngram['Observed label'] for ngram in ngrams]
        canonical_keys = [ngram['Cluster key'] for ngram in ngrams]

        self.assertEqual(observed_labels, ['agence seo', 'seo paris', 'agence seo paris'])
        self.assertEqual(canonical_keys, ['agence seo', 'paris seo', 'agence paris seo'])

    def test_order_variants_share_the_same_trigram_cluster(self):
        df = pd.DataFrame([
            {'Keyword': 'agence seo paris', 'Volume': 100},
            {'Keyword': 'seo paris agence', 'Volume': 80},
            {'Keyword': 'agence seo bordeaux', 'Volume': 50},
        ])

        clustered_df, cluster_options = add_ngram_clusters_to_view(df)

        self.assertEqual(clustered_df.iloc[0]['Cluster principal'], 'agence seo paris')
        self.assertEqual(clustered_df.iloc[1]['Cluster principal'], 'agence seo paris')
        self.assertEqual(clustered_df.iloc[2]['Cluster principal'], 'agence seo')
        self.assertEqual(clustered_df.iloc[0]['Type cluster'], 'Tri-gram')
        self.assertEqual(clustered_df.iloc[0]['Fréquence cluster'], 2)
        self.assertEqual(cluster_options[:3], ['agence seo paris', 'seo paris', 'agence seo'])

    def test_single_occurrence_ngrams_are_excluded_from_cluster_options(self):
        df = pd.DataFrame([
            {'Keyword': 'location voiture marseille', 'Volume': 120},
            {'Keyword': 'voiture location marseille', 'Volume': 110},
            {'Keyword': 'hotel bordeaux centre', 'Volume': 90},
        ])

        clustered_df, cluster_options = add_ngram_clusters_to_view(df)

        self.assertNotIn('hotel bordeaux', cluster_options)
        self.assertEqual(clustered_df.iloc[2]['Cluster principal'], '')
        self.assertEqual(clustered_df.iloc[2]['Type cluster'], 'Aucun')
        self.assertEqual(clustered_df.iloc[2]['Fréquence cluster'], 0)

    def test_custom_equivalences_are_reused_for_ngram_clustering(self):
        df = pd.DataFrame([
            {'Keyword': 'rh lyon', 'Volume': 120},
            {'Keyword': 'ressources humaines lyon', 'Volume': 80},
        ])

        clustered_df, cluster_options = add_ngram_clusters_to_view(
            df,
            custom_rules='rh=ressources humaines'
        )

        self.assertEqual(clustered_df.iloc[0]['Cluster principal'], 'rh lyon')
        self.assertEqual(clustered_df.iloc[1]['Cluster principal'], 'rh lyon')
        self.assertIn('rh lyon', cluster_options)

    def test_thematic_preset_rules_are_merged_with_custom_rules(self):
        merged_rules = merge_keyword_equivalence_rules(
            'sav=service apres vente',
            'RH / Recrutement'
        )

        self.assertIn('rh=ressources humaines', merged_rules)
        self.assertIn('sav=service apres vente', merged_rules)

    def test_visualization_summary_aggregates_top_clusters(self):
        df = pd.DataFrame([
            {
                'Keyword': 'agence seo paris',
                'Volume': 100,
                'Cluster principal': 'agence seo paris',
                'Type cluster': 'Tri-gram',
            },
            {
                'Keyword': 'seo paris agence',
                'Volume': 80,
                'Cluster principal': 'agence seo paris',
                'Type cluster': 'Tri-gram',
            },
            {
                'Keyword': 'agence seo bordeaux',
                'Volume': 250,
                'Cluster principal': 'agence seo',
                'Type cluster': 'Bi-gram',
            },
        ])

        summary_df = build_ngram_visualization_summary(df)

        self.assertEqual(len(summary_df), 2)
        self.assertEqual(summary_df.iloc[0]['Cluster principal'], 'agence seo')
        self.assertEqual(summary_df.iloc[0]['Nombre de keywords'], 1)
        self.assertEqual(summary_df.iloc[0]['Volume total'], 250)

    def test_cluster_chart_keeps_numeric_labels_as_categories(self):
        summary_df = pd.DataFrame([
            {
                'Cluster principal': '20',
                'Type cluster': 'Bi-gram',
                'Nombre de keywords': 62,
                'Volume total': 1200,
            },
            {
                'Cluster principal': '16',
                'Type cluster': 'Bi-gram',
                'Nombre de keywords': 100,
                'Volume total': 1800,
            },
        ])

        fig = create_ngram_cluster_bar_chart(summary_df)

        self.assertEqual(fig.layout.yaxis.type, 'category')
        self.assertEqual(tuple(fig.layout.yaxis.categoryarray), ('20', '16'))
        self.assertEqual(fig.layout.xaxis.title.text, 'Volume de recherche total')

    def test_original_dataframe_remains_unchanged(self):
        source_df = pd.DataFrame([
            {
                'Keyword': 'agence seo paris',
                'Volume': 100,
                'Keyword normalisé': 'agence paris seo',
            },
            {
                'Keyword': 'seo paris agence',
                'Volume': 80,
                'Keyword normalisé': 'agence paris seo',
            },
        ])

        clustered_df, _ = add_ngram_clusters_to_view(source_df)

        self.assertIn('Keyword normalisé', clustered_df.columns)
        self.assertNotIn('Cluster principal', source_df.columns)
        self.assertIn('Cluster principal', clustered_df.columns)

    def test_cluster_columns_are_forced_to_the_end(self):
        source_df = pd.DataFrame([
            {
                'Keyword': 'agence seo paris',
                'Volume': 100,
                'KD': 20,
                'client.com: Position': 5,
            },
            {
                'Keyword': 'seo paris agence',
                'Volume': 80,
                'KD': 25,
                'client.com: Position': 7,
            },
        ])

        clustered_df, _ = add_ngram_clusters_to_view(source_df)

        self.assertEqual(
            clustered_df.columns.tolist()[-3:],
            ['Cluster principal', 'Type cluster', 'Fréquence cluster']
        )


if __name__ == '__main__':
    unittest.main()
