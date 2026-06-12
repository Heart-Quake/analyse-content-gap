# Runbook, Analyse Content Gap

## Déploiement live

| Élément | Valeur |
|---|---|
| Live URL | https://space-gap.streamlit.app/ |
| Repository | `Heart-Quake/analyse-content-gap` |
| Branche | `main` |
| Entrypoint | `app.py` |
| Dernier build marker vérifié | `analyse-content-gap:998778d` |

## Commandes locales

```bash
cd /Users/vincentflaceliere/Github/analyse-content-gap
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Vérifications :

```bash
python3 -m py_compile app.py automation_seo_theme.py
python3 -m pytest
```

## Smoke test live

Dans l'iframe Streamlit `streamlitApp`, vérifier :

- `.tool-hero` count = 1 ;
- `.sidebar-logo img` count = 1 ;
- `[data-app-build]` count = 1 ;
- `--yn-bg` présent dans les styles ;
- `#2BAF9C`, `DR SEO`, `Dr. SEO` absents ;
- pas de texte `Bad message format` ;
- pas d'iframe applicatif custom non nécessaire.

## Incident connu : Bad message format

Symptôme :

- l'app alterne entre `Running` et `Connecting` ;
- une modale affiche `Bad message format` ;
- message : `Tried to use SessionInfo before it was initialized`.

Cause observée :

- micro-interaction implémentée via `streamlit.components.v1.html` pour copier du TSV ;
- le composant introduisait un iframe et du JavaScript de clipboard ;
- Streamlit Cloud pouvait recevoir un message frontend avant l'initialisation complète de session.

Correction appliquée :

- suppression de `components.html` ;
- remplacement par `st.download_button` natif ;
- commit live vérifié : `998778d`.

Règle de gouvernance :

- éviter `streamlit.components.v1.html` pour les micro-interactions non critiques ;
- préférer les composants Streamlit natifs ;
- si un composant custom devient indispensable, l'isoler et documenter son protocole.

## Dépannage courant

### Fichier refusé

Vérifier :

- extension `.csv` ;
- nom contenant `content-gap` pour le parcours principal ;
- colonnes `Keyword`, `Volume`, `KD`, `CPC`, `SERP features` ;
- au moins une colonne `: Position`.

### Domaines absents

Vérifier :

- les colonnes de position ;
- les variantes Ahrefs `: Organic Position`, normalisées automatiquement ;
- le format du fichier source.

### Résultats vides

Vérifier :

- domaine client sélectionné ;
- position maximum ;
- nombre minimum de sites ;
- filtres volume, position, cluster ou recherche keyword.

### Build Streamlit échoué

Vérifier :

- `requirements.txt` ;
- `python3 -m py_compile app.py automation_seo_theme.py` ;
- logs Streamlit Cloud ;
- absence de données client ou fichiers volumineux dans Git.

## Fichiers à ne pas commiter

- exports Ahrefs/Semrush clients ;
- `.DS_Store` ;
- caches Streamlit/Python ;
- fichiers temporaires d'analyse ;
- captures ou exports contenant des données client.
