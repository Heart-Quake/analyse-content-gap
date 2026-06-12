# Contrats de données, Analyse Content Gap

## Imports principaux

L'app accepte des fichiers CSV issus d'Ahrefs ou Semrush. Les fichiers sont lus avec plusieurs couples encodage/séparateur :

- UTF-16 + tabulation ;
- UTF-16LE + tabulation ;
- UTF-8 BOM + tabulation ;
- UTF-8 + tabulation ;
- UTF-8 BOM + virgule ;
- UTF-8 + virgule ;
- UTF-8 BOM + point-virgule ;
- UTF-8 + point-virgule.

Extension attendue : `.csv`.

Pour le parcours Content Gap principal, le nom du fichier doit contenir `content-gap`.

## Colonnes obligatoires

Colonnes génériques requises :

| Colonne | Type attendu | Rôle |
|---|---|---|
| `Keyword` | texte | mot-clé analysé |
| `Volume` | numérique | volume de recherche |
| `KD` | numérique | difficulté keyword |
| `CPC` | numérique | coût par clic |
| `SERP features` | texte | features SERP exposées par l'outil source |

Colonnes de position :

- au moins une colonne contenant `: Position` ;
- les variantes `: Organic Position` sont normalisées en `: Position` ;
- les variantes `: Organic Traffic` sont normalisées en `: Traffic`.

## Formats Ahrefs supportés

### Content Gap multi-domaines

Colonnes typiques :

```text
Keyword
Volume
KD
CPC
SERP features
client.com: URL
client.com: Organic Position
competitor.com: URL
competitor.com: Organic Position
```

Le domaine client est détecté depuis les colonnes de position. L'utilisateur le sélectionne dans la sidebar.

### Organic Keywords

Colonnes typiques :

```text
Keyword
Volume
Current URL
Current position
```

Ce format est utilisé par la vue isolée "Consolidation Ahrefs par URL".

## Formats Semrush supportés

L'app accepte les exports Organic Research > Positions si les colonnes peuvent être ramenées au contrat commun :

- `Keyword` ;
- `Volume` ;
- métriques de difficulté/CPC quand disponibles ;
- URL et position exploitables.

Si un export Semrush ne contient pas les colonnes communes, l'app doit afficher une erreur utilisateur claire plutôt que tenter une inférence silencieuse.

## Enrichissements produits

L'app ajoute des colonnes métier :

| Colonne | Rôle |
|---|---|
| `Sélection` | sélection exploitable dans le tableau |
| `Stratégie` | sauvegarde, quick win, opportunité, potentiel, conquête |
| `Marque` | classification marque / hors marque |
| `Intention` | intention de recherche |
| `Template` | contenu, produit, catégorie, autre |
| `Concurrence` | intensité concurrentielle |
| `Score opportunité` | score calculé depuis volume, KD et position |
| `Keyword normalisé` | clé de regroupement optionnelle |
| `Nb variantes regroupées` | nombre de variantes regroupées |
| `Variantes regroupées` | variantes keyword conservées pour audit |
| `Cluster principal` | cluster n-gram optionnel |
| `Type cluster` | bigram / trigram |
| `Fréquence cluster` | fréquence du cluster |

## Exports

Exports disponibles :

- CSV des résultats filtrés ;
- JSON des résultats filtrés ;
- TSV via `st.download_button`, prêt à coller dans Sheets ou Excel ;
- CSV/JSON de la vue Quick Win par URL ;
- CSV de consolidation Ahrefs par URL avec séparateur `;` et encodage `UTF-8 BOM`.

## Limites et erreurs attendues

Erreurs utilisateur fréquentes :

- fichier non CSV ;
- fichier sans `content-gap` dans le nom pour le parcours principal ;
- colonnes `Keyword` ou `Volume` absentes ;
- aucune colonne de position détectée ;
- encodage ou séparateur non détectable ;
- domaine client absent des colonnes ;
- fichier trop volumineux pour Streamlit Community Cloud.

Le code doit privilégier un message `st.error` explicite plutôt qu'un traceback brut.
