# Analyse Content Gap

Application Streamlit publique pour prioriser les opportunités SEO à partir d'exports Ahrefs ou Semrush. L'outil consolide les keywords, classe les intentions, détecte les quick wins par URL et produit des exports exploitables pour une stratégie éditoriale ou concurrentielle.

## Source live

| Élément | Valeur |
|---|---|
| Live URL | https://space-gap.streamlit.app/ |
| Repository | https://github.com/Heart-Quake/analyse-content-gap |
| Branche live | `main` |
| Entrypoint Streamlit | `app.py` |
| Commande locale | `streamlit run app.py` |
| Compilation | `python3 -m py_compile app.py automation_seo_theme.py` |
| Tests | `python3 -m pytest` |
| Build marker live vérifié | `analyse-content-gap:998778d` |
| Secrets | Aucun secret requis |
| Runtime local | fichiers uploadés en session Streamlit uniquement |

## Rôle produit

L'outil sert à analyser un gap de contenu SEO depuis des exports concurrentiels. Il répond à trois questions :

- Quels keywords sont des opportunités prioritaires ?
- Quelle stratégie appliquer : sauvegarde, quick win, opportunité, potentiel ou conquête ?
- Quelles URLs doivent être travaillées en priorité ?

Hors périmètre :

- Crawl web en direct.
- Connexion API Ahrefs ou Semrush.
- Stockage de données client.
- Génération automatique de contenus.

## Quickstart

```bash
cd /Users/vincentflaceliere/Github/analyse-content-gap
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Vérifications avant push :

```bash
python3 -m py_compile app.py automation_seo_theme.py
python3 -m pytest
```

## Documentation

- [Contrats de données](docs/DATA_CONTRACTS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Runbook Streamlit](docs/RUNBOOK.md)

## Flux fonctionnel

1. L'utilisateur importe un ou plusieurs CSV.
2. L'app lit les exports avec plusieurs encodages et séparateurs possibles.
3. Le domaine client est détecté depuis les colonnes de position.
4. Les données sont filtrées par nombre de sites concurrents et position maximum.
5. Les keywords sont enrichis avec stratégie, marque, intention, template URL et score.
6. Les résultats peuvent être filtrés, visualisés, consolidés par URL et exportés en CSV, JSON ou TSV.

## Design system

L'app doit rester alignée avec le design Automation SEO :

- `automation_seo_theme.py` doit être chargé après `st.set_page_config`.
- `logo-sidebar-cream.png` doit rester disponible à la racine.
- Le hero principal doit utiliser `.tool-hero`.
- Le marqueur caché `data-app-build` doit rester présent pour vérifier le commit live.
- Patterns interdits : `#2BAF9C`, `DR SEO`, `Dr. SEO`, `base = "light"`.

Éviter `streamlit.components.v1.html` pour les micro-interactions non critiques. Un incident live a été corrigé après un message `Bad message format / SessionInfo before it was initialized` causé par un composant HTML custom de copie presse-papiers.

## Tests couverts

Les tests existants couvrent :

- consolidation Ahrefs par URL ;
- formats d'export JSON/CSV et affichage volume ;
- normalisation/déduplication de keywords ;
- clusters n-gram.

Les tests ne couvrent pas encore :

- parcours Streamlit complet ;
- uploads très volumineux ;
- compatibilité visuelle live ;
- performances sur gros exports.

## Gouvernance Git

Ne pas commiter :

- données client ;
- exports complets Ahrefs/Semrush ;
- `.DS_Store` ;
- caches Streamlit ou Python ;
- outputs ad hoc.

Changements locaux préexistants connus au moment de cette remédiation :

- `.DS_Store`
- `Ressource/Exemple.py`
