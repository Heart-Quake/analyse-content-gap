import streamlit as st
import pandas as pd
import re
from io import BytesIO
import plotly.express as px

# Classes d'exception personnalisées
class FileProcessingError(Exception):
    """Exception levée lors d'une erreur de traitement de fichier"""
    pass

class DataValidationError(Exception):
    """Exception levée lors d'une erreur de validation des données"""
    pass

class ConfigurationError(Exception):
    """Exception levée lors d'une erreur de configuration"""
    pass

class AnalysisError(Exception):
    """Exception levée lors d'une erreur d'analyse"""
    pass

st.set_page_config(
    page_title="Analyse Content Gap",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
    <style>
    /* Hiérarchie visuelle */
    h1, h2, h3, h4, h5, h6 {
        color: #2BAF9C;  /* Vert du logo */
    }
    
    /* Style spécifique pour les titres de filtres */
    .filter-title {
        color: #2BAF9C;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Style pour les titres de sections */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #2BAF9C;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Composants interactifs - Boutons et téléchargements */
    .stButton > button, .stDownloadButton > button, div[data-testid="stSidebarNav"] button {
        background-color: #2BAF9C;  /* Vert du logo */
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stSidebarNav"] button:hover {
        background-color: #249889;  /* Version plus foncée du vert */
        transform: translateY(-1px);
    }
    
    /* Tableaux et métriques */
    .dataframe {
        font-size: 0.9rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E8E8;
    }
    .metric-container {
        background-color: #F8F9F9;
        padding: 1.25rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E8E8;
    }
    
    /* Filtres */
    .stSlider {
        padding: 1rem 0;
    }
    .stMultiSelect {
        margin-bottom: 1rem;
    }

    /* Titres des sections */
    .section-header {
        color: #249889;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Style pour les sélecteurs numériques */
    [data-testid="stNumberInput"] {
        background-color: white;
        border-radius: 0.5rem;
        border: 1px solid #E5E8E8;
        padding: 0.5rem;
    }
    [data-testid="stNumberInput"] > div {
        padding: 0.25rem;
    }
    [data-testid="stNumberInput"] input {
        font-size: 0.9rem;
    }
    [data-testid="stNumberInput"] label {
        color: #2BAF9C;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Ajustement des marges pour les colonnes */
    [data-testid="column"] {
        padding: 0.25rem !important;
    }
    </style>
""", unsafe_allow_html=True)

def process_content_gap_file(file):
    """Traite le fichier content gap et retourne un DataFrame nettoyé"""
    try:
        file.seek(0)
        df = pd.read_csv(file, sep='\t', encoding='utf-16')
        
        # Validation des colonnes requises
        required_columns = {
            'Keyword': str,
            'Volume': 'numeric',
            'KD': 'numeric',
            'CPC': 'numeric',
            'SERP features': str
        }
        
        for col, dtype in required_columns.items():
            if col not in df.columns:
                raise FileProcessingError(f"Colonne manquante : {col}")
            
            if dtype == 'numeric':
                df[col] = pd.to_numeric(
                    df[col].astype(str).replace(['', '-'], '0'), 
                    errors='coerce'
                ).fillna(0)
        
        # Validation des colonnes de position
        position_columns = [col for col in df.columns if ': Position' in col]
        if not position_columns:
            raise FileProcessingError("Aucune colonne de position trouvée")
            
        # Nettoyage des positions
        for col in position_columns:
            df[col] = pd.to_numeric(
                df[col].fillna(0).replace(['', '-'], '0'), 
                errors='coerce'
            ).fillna(0)
            
        return df

    except Exception as e:
        raise FileProcessingError(f"Erreur lors du traitement du fichier : {str(e)}")

def classify_intent(keyword: str) -> str:
    """
    Classifie l'intention de recherche du mot-clé de manière plus précise
    en utilisant des patterns et des règles de priorité
    """
    keyword = keyword.lower().strip()
    
    # Dictionnaire des patterns par type d'intention
    intent_patterns = {
        'Transactionnel': {
            'primary': [
                'acheter', 'commander', 'prix', 'tarif', 'devis',
                'cout', 'coût', 'abonnement', 'souscrire', 'payer',
                'achat', 'commande', 'reservation', 'réservation',
                'souscription', 'facture', 'facturation', 'forfait',
                'contrat', 'devis', 'offre', 'formule', 'option'
            ],
            'secondary': [
                'promotion', 'solde', 'reduction', 'réduction',
                'offre', 'pack', 'promo', 'remise', 'bon plan',
                'pas cher', 'gratuit', 'moins cher', 'meilleur prix',
                'tarification', 'mensualité', 'mensualites', 'paiement',
                'prix au m2', 'prix m2', 'prix/m2', 'prix du kwh', 
                'prix kwh', 'prix/kwh', 'prix au kwh'
            ],
            'verbs': [
                'obtenir', 'recevoir', 'demander', 'installer',
                'changer', 'remplacer', 'upgrade', 'mettre à jour',
                'mettre a jour', 'renouveler', 'signer', 'louer',
                'souscrire', 'payer', 'commander'
            ]
        },
        'Informationnel': {
            'primary': [
                'comment', 'pourquoi', 'quoi', 'que', 'quel', 'quelle',
                'quand', 'où', 'qui', 'definition', 'définition',
                'c\'est quoi', 'qu\'est-ce', 'combien', 'est-ce que',
                'peut-on', 'faut-il', 'doit-on', 'comment faire',
                'calcul', 'calcule', 'calculer'
            ],
            'secondary': [
                'guide', 'tuto', 'tutoriel', 'exemple', 'signification',
                'fonctionnement', 'explication', 'difference', 'différence',
                'conseil', 'astuce', 'methode', 'méthode', 'technique',
                'solution', 'problème', 'probleme', 'panne', 'erreur',
                'comprendre', 'savoir', 'apprendre'
            ],
            'topics': [
                'consommation', 'utilisation', 'installation',
                'maintenance', 'entretien', 'réparation', 'reparation',
                'diagnostic', 'mesure', 'estimation', 'durée', 'duree',
                'temps', 'distance', 'taille', 'dimension'
            ]
        },
        'Commercial': {
            'primary': [
                'comparatif', 'comparaison', 'vs', 'versus',
                'meilleur', 'top', 'classement', 'alternative',
                'quel fournisseur', 'quelle marque', 'quel opérateur',
                'quel prestataire', 'quelle entreprise', 'comparateur'
            ],
            'secondary': [
                'avis', 'test', 'review', 'retour', 'feedback',
                'expérience', 'experience', 'témoignage', 'temoignage',
                'recommandation', 'selection', 'sélection', 'choix'
            ],
            'comparison_terms': [
                'avantages', 'inconvénients', 'pour et contre',
                'points forts', 'points faibles', 'atouts', 'défauts',
                'plus et moins', 'positif négatif', 'forces faiblesses',
                'mieux que', 'plutôt que', 'par rapport à',
                'difference entre', 'différence entre'
            ]
        },
        'Navigationnel': {
            'primary': [
                'connexion', 'login', 'compte', 'espace client',
                'contact', 'telephone', 'téléphone', 'adresse',
                'mon compte', 'se connecter', 'inscription',
                'créer un compte', 'creer un compte', 'accès',
                'acces', 'portail'
            ],
            'secondary': [
                'service client', 'assistance', 'support',
                'agence', 'boutique', 'magasin', 'app', 
                'application', 'site', 'accueil',
                'localisation', 'horaire', 'ouverture'
            ],
            'brand_specific': [
                'officiel', 'officielle', 'france',
                'service', 'contact', 'numero',
                'siege', 'siège', 'social'
            ]
        }
    }
    
    def check_patterns(patterns: list, exact_match: bool = False) -> bool:
        if exact_match:
            return any(
                pattern == keyword or
                pattern + 's' == keyword or  # Pluriel
                pattern + 'e' == keyword or  # Féminin
                pattern + 'es' == keyword    # Féminin pluriel
                for pattern in patterns
            )
        return any(
            pattern in keyword or
            pattern + 's' in keyword or  # Pluriel
            pattern + 'e' in keyword or  # Féminin
            pattern + 'es' in keyword    # Féminin pluriel
            for pattern in patterns
        )
    
    # Vérification spéciale pour les mots-clés transactionnels
    trans_patterns = intent_patterns['Transactionnel']
    
    # 1. Vérification des patterns primaires
    if check_patterns(trans_patterns['primary']):
        return 'Transactionnel'
    
    # 2. Combinaison de patterns
    words = keyword.split()
    if len(words) >= 2:
        # Vérification des combinaisons de patterns
        has_commercial_term = check_patterns(trans_patterns['comparison_terms'])
        has_verb = check_patterns(trans_patterns['verbs'])
        has_secondary = check_patterns(trans_patterns['secondary'])
        
        # Si on a une combinaison de termes commerciaux et de verbes
        if (has_commercial_term and has_verb) or \
           (has_commercial_term and has_secondary) or \
           (has_verb and has_secondary):
            return 'Transactionnel'
    
    # Vérification des autres intentions
    for intent, patterns in intent_patterns.items():
        if intent == 'Transactionnel':
            continue
            
        # Vérification des patterns primaires
        if check_patterns(patterns['primary']):
            return intent
            
        # Vérification des patterns secondaires
        if check_patterns(patterns['secondary']):
            if len(keyword.split()) >= 2:  # Au moins deux mots pour confirmer l'intention
                return intent
    
    # Si le mot-clé est un seul mot et ne correspond à aucun pattern
    if len(keyword.split()) == 1:
        return 'Navigationnel'
    
    # Si aucun pattern n'a été trouvé
    return 'Autre'

def classify_branded(keyword: str, client_name: str, custom_brands: list) -> str:
    """Détermine si le mot-clé est de marque ou non"""
    keyword = keyword.lower()
    
    # Ajout du nom du client et des termes personnalisés à la liste des marques
    brands = [client_name.lower()] + [brand.lower() for brand in custom_brands if brand]
    
    # Vérification si le mot-clé contient une marque
    if any(brand in keyword for brand in brands if brand):
        return 'Marque'
    return 'Hors marque'

def display_filtered_results(filtered_df, client_name):
    """Affiche les résultats filtrés"""
    try:
        client_domain = clean_domain_name(client_name)
        
        with st.expander("🔍 Filtres avancés", expanded=True):
            # Création de 3 colonnes pour les filtres principaux
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # Stratégie
                st.markdown("""
                    <h6 style='color: #2BAF9C; margin-bottom: 0px; padding-bottom: 0px;'>
                        Stratégie
                    </h6>
                """, unsafe_allow_html=True)
                strategies = [
                    'Sauvegarde', 'Quick Win', 'Opportunité',
                    'Potentiel', 'Conquête', 'Non positionné'
                ]
                selected_strategies = st.multiselect(
                    "",
                    options=strategies,
                    default=strategies,
                    key='strategy_filter'
                )
            
            with col2:
                # Marque
                st.markdown("""
                    <h6 style='color: #2BAF9C; margin-bottom: 0px; padding-bottom: 0px;'>
                        Marque
                    </h6>
                """, unsafe_allow_html=True)
                brand_options = ['Marque', 'Hors marque']
                selected_brands = st.multiselect(
                    "",
                    options=brand_options,
                    default=brand_options,
                    key='brand_filter'
                )
            
            with col3:
                # Intention
                st.markdown("""
                    <h6 style='color: #2BAF9C; margin-bottom: 0px; padding-bottom: 0px;'>
                        Intention
                    </h6>
                """, unsafe_allow_html=True)
                intent_options = ['Transactionnel', 'Informationnel', 'Commercial', 'Navigationnel', 'Autre']
                selected_intentions = st.multiselect(
                    "",
                    options=intent_options,
                    default=intent_options,
                    key='intent_filter'
                )

            # Création de 2 colonnes pour les métriques
            metric_col1, metric_col2 = st.columns([1, 1])
            
            with metric_col1:
                # Volume
                st.markdown("""
                    <h6 style='color: #2BAF9C; margin-bottom: 0px; padding-bottom: 0px;'>
                        Volume
                    </h6>
                """, unsafe_allow_html=True)
                vol_col1, vol_col2 = st.columns(2)
                with vol_col1:
                    vol_min = int(filtered_df['Volume'].min())
                    volume_min = st.number_input("Min", min_value=vol_min, value=vol_min, key='vol_min')
                with vol_col2:
                    vol_max = int(filtered_df['Volume'].max())
                    volume_max = st.number_input("Max", max_value=vol_max, value=vol_max, key='vol_max')
            
            with metric_col2:
                # Position
                st.markdown("""
                    <h6 style='color: #2BAF9C; margin-bottom: 0px; padding-bottom: 0px;'>
                        Position
                    </h6>
                """, unsafe_allow_html=True)
                pos_col1, pos_col2 = st.columns(2)
                with pos_col1:
                    position_min = st.number_input("Min", min_value=0, value=0, key='pos_min')
                with pos_col2:
                    position_max = st.number_input(
                        "Max",
                        value=int(filtered_df[f'{client_domain}: Position'].max()),
                        key='pos_max'
                    )

        # Application des filtres
        filtered_df = filtered_df[
            (filtered_df['Stratégie'].isin(selected_strategies)) &
            (filtered_df['Marque'].isin(selected_brands)) &
            (filtered_df['Intention'].isin(selected_intentions)) &
            (filtered_df['Volume'].between(volume_min, volume_max)) &
            (filtered_df[f'{client_domain}: Position'].fillna(0).between(position_min, position_max))
        ]

        # Affichage des métriques
        st.markdown('<p class="subheader">📈 Synthèse des mots-clés</p>', unsafe_allow_html=True)
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("🎯 Total mots-clés", f"{len(filtered_df):,}")
        with metric_col2:
            st.metric("📊 Volume total", f"{int(filtered_df['Volume'].sum()):,}")
        with metric_col3:
            st.metric("📈 KD moyen", f"{round(filtered_df['KD'].mean(), 1)}")
        with metric_col4:
            st.metric("💰 CPC moyen", f"${round(filtered_df['CPC'].mean(), 2)}")

        # Création des onglets
        tab1, tab2, tab3 = st.tabs(["📊 Résultats filtrés", "📈 Répartition par stratégie", "🔍 Visualisations"])
        
        with tab1:
            st.dataframe(filtered_df, use_container_width=True, height=400)
            export_data(filtered_df, client_name)

        with tab2:
            display_strategy_stats(filtered_df)

        with tab3:
            display_visualizations(filtered_df, client_name)

    except Exception as e:
        st.error(f"❌ Erreur lors de l'affichage des résultats : {str(e)}")
        st.info("🔄 Essayez de relancer l'analyse")

def process_and_store_data(uploaded_files, client_name, nombre_sites, top_position, custom_brands):
    """Traitement des données avec validation améliorée"""
    try:
        if not uploaded_files:
            raise FileProcessingError("Aucun fichier n'a été téléchargé")
            
        # Traitement du fichier content gap
        for uploaded_file in uploaded_files:
            try:
                # Validation du format du fichier
                is_valid, error_msg = validate_file_format(uploaded_file.name)
                if not is_valid:
                    raise FileProcessingError(error_msg)

                # Lecture et traitement initial du fichier
                df = process_content_gap_file(uploaded_file)
                
                # Nettoyage des noms de domaines dans les en-têtes
                rename_dict = {}
                for col in df.columns:
                    if ': Position' in col or ': Traffic' in col or ': URL' in col:
                        domain = col.split(':')[0]
                        clean_domain = clean_domain_name(domain)
                        if domain != clean_domain:
                            new_col = col.replace(domain, clean_domain)
                            rename_dict[col] = new_col
                
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                
                # Filtrage des résultats
                df_filtered = filter_results(df, nombre_sites, top_position)
                
                # Ajout des colonnes d'analyse
                df_final = enrich_data(df_filtered, client_name, custom_brands)
                
                # Stockage des résultats
                st.session_state.df_final = df_final
                st.session_state.analysis_done = True
                
                st.success(f"✅ Fichier traité avec succès : {uploaded_file.name}")
                return True
                
            except Exception as e:
                st.warning(f"⚠️ Erreur lors du traitement de {uploaded_file.name}: {str(e)}")
                continue

        raise AnalysisError("Aucun fichier n'a pu être traité correctement")

    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse : {str(e)}")
        return False

def filter_results(df, nombre_sites, top_position):
    """
    Filtre les résultats pour trouver les mots-clés où au moins X concurrents 
    sont positionnés dans le top Y des résultats Google
    
    Args:
        df: DataFrame avec les données
        nombre_sites: Nombre minimum de sites concurrents requis
        top_position: Position maximum à considérer
    """
    try:
        # Identification des colonnes de position
        position_columns = [col for col in df.columns if ': Position' in col]
        
        # Création d'une matrice booléenne pour les positions valides
        # Une position est valide si elle est > 0 ET <= top_position
        positions_valides = df[position_columns].apply(
            lambda x: (x > 0) & (x <= top_position)
        )
        
        # Compte le nombre de concurrents positionnés dans le top Y pour chaque mot-clé
        nb_concurrents_top_y = positions_valides.sum(axis=1)
        
        # Filtre les mots-clés où au moins X concurrents sont dans le top Y
        df_filtered = df[nb_concurrents_top_y >= nombre_sites]
        
        if len(df_filtered) == 0:
            raise DataValidationError(
                f"Aucun mot-clé n'a au moins {nombre_sites} "
                f"concurrent(s) dans le top {top_position}"
            )
            
        return df_filtered
        
    except Exception as e:
        raise AnalysisError(f"Erreur lors du filtrage : {str(e)}")

def enrich_data(df, client_name, custom_brands):
    """Enrichit les données avec des colonnes d'analyse"""
    try:
        # Nettoyage du nom de domaine client
        client_domain = clean_domain_name(client_name)
        
        # Ajout des colonnes d'analyse dans l'ordre correct
        df['Sélection'] = ''
        df['Stratégie'] = df.apply(lambda row: define_strategy(row, client_domain), axis=1)
        df['Marque'] = df['Keyword'].apply(lambda x: classify_branded(x, client_domain, custom_brands))
        df['Intention'] = df['Keyword'].apply(classify_intent)
        
        # Calcul de la concurrence
        position_columns = [col for col in df.columns if ': Position' in col]
        df['Concurrence'] = df[position_columns].apply(lambda x: (x > 0).sum(), axis=1)
        
        # Réorganisation des colonnes
        column_order = [
            'Keyword', 'Sélection', 'Stratégie', 'Marque', 'Intention',
            'Concurrence', 'Volume', 'KD', 'CPC', 'SERP features'
        ]
        
        # Ajout des colonnes spécifiques aux domaines
        for suffix in [': Position', ': URL', ': Traffic']:
            client_cols = [col for col in df.columns if client_domain + suffix in col]
            other_cols = [col for col in df.columns if suffix in col and client_domain not in col]
            column_order.extend(client_cols + sorted(other_cols))
        
        return df[column_order]
        
    except Exception as e:
        raise DataValidationError(f"Erreur lors de l'enrichissement des données : {str(e)}")

def define_strategy(row, client_name):
    """Définit la stratégie SEO pour chaque mot-clé"""
    try:
        # Nettoyage du nom de domaine client
        client_domain = clean_domain_name(client_name)
        
        # Initialisation des variables
        client_position = row.get(f"{client_domain}: Position", 0)
        
        # Définition de la stratégie selon la position
        if client_position == 1:
            return "Sauvegarde"
        elif 2 <= client_position <= 5:
            return "Quick Win"
        elif 6 <= client_position <= 10:
            return "Opportunité"
        elif 11 <= client_position <= 20:
            return "Potentiel"
        elif client_position > 20:
            return "Conquête"
        else:
            return "Non positionné"
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la définition de la stratégie : {str(e)}")
        return "Non défini"

def calculate_opportunity_score(volume, kd, position):
    """Calcule le score d'opportunité pour un mot-clé"""
    try:
        # Normalisation du volume (0-100)
        volume_score = min(volume / 1000, 100)
        
        # Normalisation de la difficulté (0-100)
        difficulty_score = 100 - min(kd, 100)
        
        # Normalisation de la position (0-100)
        position_score = 100 - min(position if position > 0 else 100, 100)
        
        # Calcul du score final (moyenne pondérée)
        final_score = (volume_score * 0.4) + (difficulty_score * 0.3) + (position_score * 0.3)
        
        return round(final_score, 2)
        
    except Exception as e:
        st.warning(f"⚠️ Erreur lors du calcul du score : {str(e)}")
        return 0

def display_strategy_stats(filtered_df):
    """Affiche les statistiques par stratégie"""
    try:
        # Définir l'ordre des stratégies
        strategy_order = [
            'Sauvegarde',    # Position 1
            'Quick Win',     # Positions 2-5
            'Opportunité',   # Positions 6-10
            'Potentiel',     # Positions 11-20
            'Conquête',      # Positions > 20
            'Non positionné' # Absent du top 100
        ]
        
        # Calcul des statistiques
        stats_df = filtered_df.groupby('Stratégie').agg({
            'Keyword': 'count',
            'Volume': 'sum',
            'KD': 'mean',
            'CPC': 'mean'
        })
        
        # Réorganisation selon l'ordre défini
        stats_df = stats_df.reindex(strategy_order)
        
        # Renommage et formatage des colonnes
        stats_df.columns = ['Nombre de mots-clés', 'Volume total', 'KD moyen', 'CPC moyen']
        
        # Formatage des valeurs
        stats_df['Nombre de mots-clés'] = stats_df['Nombre de mots-clés'].fillna(0).astype(int)
        stats_df['Volume total'] = stats_df['Volume total'].fillna(0).astype(int).apply(lambda x: f"{x:,}")
        stats_df['KD moyen'] = stats_df['KD moyen'].round(1)
        stats_df['CPC moyen'] = stats_df['CPC moyen'].round(2).apply(lambda x: f"${x:.2f}")
        
        # Calcul des totaux
        totals = pd.Series({
            'Nombre de mots-clés': stats_df['Nombre de mots-clés'].sum(),
            'Volume total': stats_df['Volume total'].str.replace(',', '').astype(int).sum(),
            'KD moyen': stats_df['KD moyen'].mean().round(1),
            'CPC moyen': f"${stats_df['CPC moyen'].str.replace('$', '').astype(float).mean():.2f}"
        }, name='Total')
        
        # Ajout de la ligne des totaux
        stats_df = pd.concat([stats_df, pd.DataFrame([totals])])
        
        # Affichage avec style
        st.markdown("### 📊 Répartition par stratégie")
        
        # Création du style pour le tableau
        styles = [
            dict(selector="th", props=[("font-size", "1.1em"), 
                                     ("text-align", "center"),
                                     ("background-color", "#f0f2f6"),
                                     ("color", "#2BAF9C"),
                                     ("font-weight", "bold"),
                                     ("padding", "12px")]),
            dict(selector="td", props=[("text-align", "center"),
                                     ("padding", "8px")]),
            dict(selector="tr:last-child", props=[("font-weight", "bold"),
                                                ("background-color", "#f0f2f6")])
        ]
        
        # Application du style et affichage
        st.dataframe(
            stats_df,
            use_container_width=True,
            height=250,
            column_config={
                "Nombre de mots-clés": st.column_config.NumberColumn(format="%d"),
                "Volume total": st.column_config.TextColumn(),
                "KD moyen": st.column_config.NumberColumn(format="%.1f"),
                "CPC moyen": st.column_config.TextColumn()
            }
        )
        
    except Exception as e:
        st.error(f"❌ Erreur lors de l'affichage des statistiques : {str(e)}")

def display_visualizations(filtered_df, client_name):
    """Affiche les visualisations"""
    col1, col2, col3 = st.columns(3)
    
    # Configuration commune pour tous les graphiques
    graph_title_style = {
        'font': {'size': 16, 'family': 'Arial'},
        'y': 0.95
    }
    graph_layout = {
        'title_font': graph_title_style['font'],
        'showlegend': True,
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white',
        'margin': dict(t=50, b=30, l=30, r=30)
    }
    
    with col1:
        # Distribution des volumes par position
        fig_volume = create_position_volume_histogram(filtered_df, client_name)
        fig_volume.update_layout(
            title=dict(
                text='Distribution du volume de recherche par position',
                **graph_title_style
            ),
            **graph_layout
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        # Répartition des stratégies
        strategy_order = ['Sauvegarde', 'Quick Win', 'Opportunité', 'Potentiel', 'Conquête', 'Non positionné']
        fig_strategies = px.pie(
            filtered_df,
            names='Stratégie',
            title='Répartition des stratégies',
            category_orders={'Stratégie': strategy_order}
        )
        fig_strategies.update_layout(
            title=dict(
                text='Répartition des stratégies',
                **graph_title_style
            ),
            **graph_layout
        )
        st.plotly_chart(fig_strategies, use_container_width=True)
    
    with col3:
        # Répartition des intentions
        fig_intentions = px.pie(
            filtered_df,
            names='Intention',
            title='Répartition des intentions de recherche'
        )
        fig_intentions.update_layout(
            title=dict(
                text='Répartition des intentions de recherche',
                **graph_title_style
            ),
            **graph_layout
        )
        st.plotly_chart(fig_intentions, use_container_width=True)

def export_data(filtered_df, client_name):
    """Export des données au format CSV"""
    st.download_button(
        "📥 Export CSV",
        filtered_df.to_csv(index=False),
        f"Analyse_Concurrentielle_{client_name}.csv",
        mime="text/csv",
        help="Télécharger les résultats au format CSV",
        type="primary"
    )

def create_position_volume_histogram(filtered_df, client_name):
    """Crée un histogramme de distribution des volumes par position"""
    # Création des catégories de position
    def categorize_position(pos):
        if pd.isnull(pos) or pos == 0:
            return 'Non positionné'
        elif pos <= 3:
            return 'Top 3'
        elif pos <= 10:
            return 'Top 4-10'
        elif pos <= 20:
            return 'Top 11-20'
        else:
            return 'Top 21-100'

    # Ajout de la catégorie de position
    df_viz = filtered_df.copy()
    df_viz['Position_Category'] = df_viz[f'{client_name}: Position'].apply(categorize_position)
    
    # Calcul de la somme des volumes par catégorie avec le nom de colonne correct
    position_volume = df_viz.groupby('Position_Category')['Volume'].sum().reset_index()
    
    # Définir l'ordre personnalisé des catégories
    category_order = ['Top 3', 'Top 4-10', 'Top 11-20', 'Top 21-100', 'Non positionné']
    position_volume['Position_Category'] = pd.Categorical(
        position_volume['Position_Category'],
        categories=category_order,
        ordered=True
    )
    
    # Création du graphique
    fig_volume = px.bar(
        position_volume.sort_values('Position_Category'),
        x='Position_Category',
        y='Volume',
        title='Distribution du volume de recherche par position',
        labels={
            'Position_Category': 'Position',
            'Volume': 'Volume de recherche'
        }
    )
    
    # Personnalisation du graphique
    fig_volume.update_layout(
        title_font_size=20,
        title_font_family="Arial",
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.2,
        margin=dict(t=50, b=50, l=50, r=25),
        xaxis=dict(
            title_font_size=14,
            tickfont_size=12,
            gridcolor='#E5E8E8'
        ),
        yaxis=dict(
            title_font_size=14,
            tickfont_size=12,
            gridcolor='#E5E8E8'
        )
    )
    
    # Couleurs personnalisées
    fig_volume.update_traces(
        marker_color=['#2ECC71', '#3498DB', '#F1C40F', '#E67E22', '#E74C3C']
    )
    
    return fig_volume

def add_contextual_help():
    """Ajoute des explications contextuelles dans l'interface"""
    with st.expander("ℹ️ Comment utiliser cet outil ?", expanded=False):
        st.markdown("""
        ### Processus en 4 étapes :

        1. **Préparation des données**
           - Exportez les données de vos outils SEO
           - Assurez-vous d'avoir les fichiers pour chaque concurrent
           - Nommez les fichiers avec le domaine (ex: monsite.csv)

        2. **Import et configuration**
           - Importez tous vos fichiers en une fois
           - Sélectionnez votre domaine client
           - Ajustez les paramètres d'analyse selon vos besoins

        3. **Analyse des résultats**
           - Utilisez les filtres pour affiner votre analyse
           - Examinez les différentes visualisations
           - Identifiez les opportunités prioritaires

        4. **Export et action**
           - Exportez les résultats filtrés
           - Utilisez les données pour votre stratégie SEO
           - Suivez l'évolution des positions
        """)

def add_metric_explanations():
    """Ajoute des explications pour chaque métrique"""
    with st.expander("📊 Comprendre les métriques", expanded=False):
        st.markdown("""
        ### Métriques principales

        #### 🎯 Stratégie
        - **Sauvegarde** : Mots-clés en position 1 - Focus sur la défense
        - **Quick Win** : Positions 2-5 - Potentiel de gain rapide
        - **Opportunité** : Positions 6-10 - Progression possible
        - **Potentiel** : Positions 11-20 - Travail à moyen terme
        - **Conquête** : Positions > 20 - Objectif long terme

        #### 📈 Métriques SEO
        - **Volume** : Nombre moyen de recherches mensuelles
        - **KD** : Score de difficulté (0-100)
        - **Position** : Position actuelle dans les résultats
        - **CPC** : Coût par clic moyen
        """)

def initialize_session_state():
    """Initialise les variables de session"""
    if 'df_final' not in st.session_state:
        st.session_state.df_final = None
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

def clean_domain_name(domain: str) -> str:
    """Nettoie le nom de domaine en retirant www. et les slashes"""
    domain = domain.lower().strip()
    # Suppression du protocole (http:// ou https://)
    if '://' in domain:
        domain = domain.split('://')[-1]
    # Suppression du www.
    if domain.startswith('www.'):
        domain = domain[4:]
    # Suppression des slashes
    domain = domain.rstrip('/')
    return domain

def extract_domains_from_files(uploaded_files):
    """Extrait les domaines des fichiers téléchargés"""
    try:
        if not uploaded_files:
            return [], None
            
        # Lecture et traitement du fichier
        df = process_content_gap_file(uploaded_files[0])
        
        # Extraction des domaines depuis les colonnes de position
        domains = set()
        for col in df.columns:
            if ': Position' in col:
                domain = col.split(':')[0]
                clean_domain = clean_domain_name(domain)
                domains.add(clean_domain)
                
        return sorted(list(domains)), df
                
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction des domaines : {str(e)}")
        return [], None

def validate_file_format(file_name: str) -> tuple[bool, str]:
    """Valide le format du fichier"""
    # Nettoyer le nom du fichier en retirant les tirets au début
    clean_name = file_name.lstrip('-')
    
    if not clean_name.endswith('.csv'):
        return False, "Le fichier doit être au format CSV"
    if "content-gap" not in clean_name:
        return False, "Le fichier doit être un export content gap"
    return True, ""

def main():
    """Fonction principale de l'application"""
    try:
        initialize_session_state()
        
        # En-tête avec description
        st.title("🔍 Analyse Content Gap")
        
        # Guide d'utilisation en dropdown dans la zone principale
        with st.expander("📖 Guide d'utilisation", expanded=False):
            st.markdown("""
            ## 1. Import des données
            ### Formats acceptés :
            - **Ahrefs** : `domain-organic-keywords.csv`
               - Export depuis : Organic Keywords > Export
               - Encodage : UTF-16
            - **Semrush** : `domain-organic.Positions.csv`
               - Export depuis : Organic Research > Positions
               - Encodage : UTF-8
            """)
        
        # Configuration dans la sidebar
        with st.sidebar:
            st.image("DR SEO Header.svg", use_column_width=True)
            st.header("⚙️ Configuration")
            
            uploaded_files = st.file_uploader(
                "📤 Importer les fichiers CSV",
                accept_multiple_files=True,
                type=['csv'],
                help="Formats acceptés : Ahrefs (organic-keywords.csv) ou Semrush (organic.Positions.csv)"
            )

            if uploaded_files:
                try:
                    domains, df = extract_domains_from_files(uploaded_files)
                    if df is None:
                        st.warning("⚠️ Aucun domaine n'a pu être extrait des noms de fichiers")
                    else:
                        client_name = st.selectbox(
                            "🎯 Sélectionner le client",
                            options=domains,
                            help="Sélectionnez le domaine correspondant au client"
                        )
                        
                        # Termes de marque
                        brand_terms = st.text_input(
                            "🔤 Termes de marque (séparés par des virgules)",
                            help="Ajoutez des variations de la marque à surveiller"
                        )
                        custom_brands = [t.strip() for t in brand_terms.split(',')] if brand_terms else []
                        
                        # Paramètres avancés
                        st.subheader("📊 Paramètres d'analyse")
                        nombre_sites = st.number_input(
                            "Nombre minimum de sites", 
                            min_value=1, 
                            value=1,
                            help="Nombre minimum de sites concurrents positionnés"
                        )
                        top_position = st.number_input(
                            "Position maximum", 
                            min_value=1, 
                            value=20,
                            help="Position maximum à prendre en compte"
                        )

                        # Bouton d'action
                        st.markdown("---")
                        if client_name:
                            if st.button("🚀 Lancer l'analyse", type="primary"):
                                if process_and_store_data(uploaded_files, client_name, nombre_sites, top_position, custom_brands):
                                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Erreur lors de la configuration : {str(e)}")
            else:
                st.info("ℹ️ Commencez par importer vos fichiers d'analyse")

        # Affichage des résultats si disponibles
        if st.session_state.analysis_done and st.session_state.df_final is not None:
            try:
                display_filtered_results(st.session_state.df_final, client_name)
            except Exception as e:
                st.error(f"❌ Erreur lors de l'affichage des résultats : {str(e)}")
                st.info("🔄 Essayez de relancer l'analyse")

    except Exception as e:
        st.error(f"❌ Erreur système critique : {str(e)}")
        st.info("""
        🔧 Solutions possibles :
        1. Rafraîchissez la page
        2. Vérifiez vos fichiers d'entrée
        3. Contactez le support technique si l'erreur persiste
        """)

if __name__ == "__main__":
    main() 