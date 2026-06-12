# Architecture, Analyse Content Gap

## Vue d'ensemble

L'application est actuellement concentrée dans `app.py`. Ce fichier porte à la fois :

- l'interface Streamlit ;
- la lecture des exports ;
- les règles de normalisation ;
- l'enrichissement SEO ;
- les visualisations ;
- les exports.

La priorité documentaire est donc de rendre les zones fonctionnelles lisibles sans devoir parcourir tout le fichier.

## Flux principal

```text
upload CSV
  -> process_content_gap_file
  -> extract_domains_from_files
  -> sidebar configuration
  -> process_and_store_data
  -> filter_results
  -> enrich_data
  -> display_filtered_results
  -> export_data / visualizations / quick wins
```

## Blocs métier

### Lecture et validation

- `process_content_gap_file` teste les encodages et séparateurs supportés.
- `validate_file_format` vérifie l'extension et le nom du fichier.
- `extract_domains_from_files` détecte les domaines disponibles depuis les colonnes de position.

### Consolidation Ahrefs par URL

- `detect_ahrefs_consolidation_columns` supporte les formats `Current URL / Current position` et `<domaine>: URL / <domaine>: Organic Position`.
- `process_ahrefs_consolidation_file` charge et prépare l'export.
- `build_ahrefs_url_consolidation` agrège par URL, top keyword, volume total et position moyenne.

### Normalisation keyword

- `normalize_search_text` normalise les chaînes pour la recherche.
- `build_keyword_equivalence_map` lit les équivalences custom.
- `merge_keyword_equivalence_rules` combine preset et règles utilisateur.
- `normalize_keyword_for_grouping` construit la clé de déduplication.
- `deduplicate_keywords_by_normalized_key` conserve la variante la plus utile, en priorisant volume, KD et position client.

### Clusters n-gram

- `normalize_keyword_for_ngrams` prépare les tokens.
- `extract_keyword_ngrams` génère bigrams/trigrams.
- `build_ngram_cluster_catalog` calcule les clusters fréquents.
- `add_ngram_clusters_to_view` enrichit la vue filtrée.
- `create_ngram_cluster_bar_chart` produit le graphe de synthèse.

### Classification SEO

- `classify_intent` détecte l'intention.
- `classify_branded` distingue marque / hors marque.
- `define_template` estime le template d'URL.
- `define_strategy` classe les opportunités par stratégie.
- `calculate_opportunity_score` calcule le score d'opportunité.

### Affichage et exports

- `display_filtered_results` orchestre les tabs résultats, stats, visualisations et quick wins.
- `display_results_dataframe` applique le rendu dataframe.
- `display_quick_win_by_url` regroupe les quick wins par URL.
- `export_data` expose CSV/JSON.
- `render_copy_to_clipboard_button` expose désormais un téléchargement TSV natif.

## État Streamlit

Clés principales :

- `df_final` : DataFrame enrichi final.
- `analysis_done` : booléen indiquant qu'une analyse est disponible.
- `enable_ngram_clustering` : activation des clusters n-gram.
- `ngram_cluster_filter` : filtre cluster sélectionné.
- `keyword_equivalence_rules` : règles utilisateur.
- `keyword_equivalence_preset` : preset sélectionné.

## Design system

L'entrypoint doit respecter l'ordre :

```python
st.set_page_config(...)
apply_automation_seo_theme()
```

Le thème injecte :

- variables `--yn-*` ;
- logo sidebar ;
- build marker `data-app-build` ;
- styles des contrôles Streamlit.

## Dette technique identifiée

- `app.py` est monolithique et devrait être découpé à terme.
- Les contrats Ahrefs/Semrush devraient être extraits dans un module dédié.
- Les fonctions d'UI et les fonctions métier sont mélangées.
- Les tests couvrent des fonctions métier mais pas le parcours Streamlit complet.
