from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import threading
from dotenv import load_dotenv
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)

from utils.similarity_engine import SimilarityEngine, calculate_feature_weights
from utils.data_host import DataHostManager
from utils.data_loader import DataLoader
from difflib import get_close_matches
import unicodedata
import pandas as pd

def safe_get(player_row, col_name, default=0):
    val = player_row.get(col_name, default)
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_quantile(df, col_name, q, default=0):
    if col_name not in df.columns:
        return default
    val = df[col_name].quantile(q)
    return val if pd.notna(val) else default

def normalize_player_name(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))

def determine_archetypes(player_row, df, position):
    archetypes = []
    
    if df is None or len(df) == 0:
        return archetypes
    
    if position == 'F':
        goals = safe_get(player_row, 'I_F_goals', 0)
        primary_assists = safe_get(player_row, 'I_F_primaryAssists', 0)
        secondary_assists = safe_get(player_row, 'I_F_secondaryAssists', 0)
        points = safe_get(player_row, 'I_F_points', 0)
        shot_attempts = safe_get(player_row, 'I_F_shotAttempts', 0)
        hits = safe_get(player_row, 'I_F_hits', 0)
        takeaways = safe_get(player_row, 'I_F_takeaways', 0)
        blocked_shots = safe_get(player_row, 'shotsBlockedByPlayer', 0)
        goals_against = safe_get(player_row, 'OnIce_A_goals', 0)
        penalty_minutes = safe_get(player_row, 'penaltyMinutes', 0)
        height = safe_get(player_row, 'height', 0)
        weight = safe_get(player_row, 'weight', 0)
        
        if len(df) == 0:
            return archetypes
        
        total_assists = primary_assists + secondary_assists
        assists_to_goals = total_assists / max(goals, 0.1)
        
        goals_p75 = safe_quantile(df, 'I_F_goals', 0.75)
        assists_p75 = safe_quantile(df, 'I_F_primaryAssists', 0.75)
        points_p75 = safe_quantile(df, 'I_F_points', 0.75)
        points_p60 = safe_quantile(df, 'I_F_points', 0.60)
        shot_attempts_p75 = safe_quantile(df, 'I_F_shotAttempts', 0.75)
        hits_p75 = safe_quantile(df, 'I_F_hits', 0.75)
        hits_p60 = safe_quantile(df, 'I_F_hits', 0.60)
        takeaways_p75 = safe_quantile(df, 'I_F_takeaways', 0.75)
        goals_against_p60 = safe_quantile(df, 'OnIce_A_goals', 0.60)
        blocked_p75 = safe_quantile(df, 'shotsBlockedByPlayer', 0.75)
        penalty_minutes_p60 = safe_quantile(df, 'penaltyMinutes', 0.60)
        
        if goals >= goals_p75 and shot_attempts >= shot_attempts_p75:
            archetypes.append('Sniper')
        
        if total_assists >= assists_p75 and points >= points_p75 and assists_to_goals >= 1.2 and total_assists > goals:
            archetypes.append('Playmaker')
        
        if (hits >= hits_p60) and points >= points_p60 and (height >= 72 and weight >= 200):
            archetypes.append('Power Forward')
        
        if ((blocked_shots >= blocked_p75) or (takeaways >= takeaways_p75)) and points < points_p60:
            archetypes.append('Defensive Forward')
        
        if points >= points_p60 and goals_against <= goals_against_p60 and (takeaways >= takeaways_p75 or blocked_shots >= blocked_p75):
            archetypes.append('Two-Way')
        
        if hits >= hits_p75 and points < points_p60 and penalty_minutes >= penalty_minutes_p60:
            archetypes.append('Grinder')
    
    elif position == 'D':
        goals = safe_get(player_row, 'I_F_goals', 0)
        primary_assists = safe_get(player_row, 'I_F_primaryAssists', 0)
        secondary_assists = safe_get(player_row, 'I_F_secondaryAssists', 0)
        points = safe_get(player_row, 'I_F_points', 0)
        shot_attempts = safe_get(player_row, 'I_F_shotAttempts', 0)
        hits = safe_get(player_row, 'I_F_hits', 0)
        blocked_shots = safe_get(player_row, 'shotsBlockedByPlayer', 0)
        goals_against = safe_get(player_row, 'OnIce_A_goals', 0)
        penalty_minutes = safe_get(player_row, 'penaltyMinutes', 0)
        corsiPercentage = safe_get(player_row, 'onIce_corsiPercentage', 0)
        takeaways = safe_get(player_row, 'I_F_takeaways', 0)
        pk_minutes = safe_get(player_row, 'timeOnIcePK', 0)
        
        if len(df) == 0:
            return archetypes
        
        total_assists = primary_assists + secondary_assists
        
        points_p75 = safe_quantile(df, 'I_F_points', 0.75)
        points_p45 = safe_quantile(df, 'I_F_points', 0.45)
        goals_p75 = safe_quantile(df, 'I_F_goals', 0.75)
        assists_p75 = safe_quantile(df, 'I_F_primaryAssists', 0.75)
        shot_attempts_p75 = safe_quantile(df, 'I_F_shotAttempts', 0.75)
        hits_p75 = safe_quantile(df, 'I_F_hits', 0.75)
        blocked_p75 = safe_quantile(df, 'shotsBlockedByPlayer', 0.75)
        blocked_p55 = safe_quantile(df, 'shotsBlockedByPlayer', 0.55)
        goals_against_p40 = safe_quantile(df, 'OnIce_A_goals', 0.40)
        corsiPercentage_p55 = safe_quantile(df, 'onIce_corsiPercentage', 0.55)
        corsiPercentage_p40 = safe_quantile(df, 'onIce_corsiPercentage', 0.40)
        penalty_minutes_p50 = safe_quantile(df, 'penaltyMinutes', 0.50)
        takeaways_p75 = safe_quantile(df, 'I_F_takeaways', 0.75)
        takeaways_p95 = safe_quantile(df, 'I_F_takeaways', 0.95)
        blocked_p95 = safe_quantile(df, 'shotsBlockedByPlayer', 0.95)
        pk_minutes_p35 = safe_quantile(df, 'timeOnIcePK', 0.35)
        points_p35 = safe_quantile(df, 'I_F_points', 0.35)
        
        if shot_attempts >= shot_attempts_p75 and goals >= goals_p75:
            archetypes.append('Point Shooter')
        
        if points >= points_p75 and (total_assists >= assists_p75):
            archetypes.append('Quarterback')
        
        if (
            points >= points_p45 and
            goals_against <= goals_against_p40 and
            corsiPercentage >= corsiPercentage_p55 and
            (
                takeaways >= takeaways_p95 or
                blocked_shots >= blocked_p95 or
                (pk_minutes >= pk_minutes_p35 and
                (takeaways >= takeaways_p75 or blocked_shots >= blocked_p75))
            )
        ):
            archetypes.append("Two-Way")
        
        if hits >= hits_p75 and points < points_p35 and penalty_minutes >= penalty_minutes_p50:
            archetypes.append('Grinder')
        
        if blocked_shots >= blocked_p75:
            archetypes.append('Shot Blocker')
        
        if blocked_shots >= blocked_p55 and corsiPercentage <= corsiPercentage_p40 and points < points_p35:
            archetypes.append('Stay-at-Home')
    
    return archetypes

app = Flask(__name__)
CORS(app)

cache = {
    'forwards': None,
    'defensemen': None,
    'forwards_similarity': None,
    'defensemen_similarity': None,
    'loaded': False,
}
loading_state = {'in_progress': False, 'error': None, 'thread': None}
data_host = DataHostManager()

def find_player_in_dataframe(df, player_name, season):
    if player_name is None:
        return None
    player_name_lower = str(player_name).lower().strip()
    player_name_norm = normalize_player_name(player_name)
    
    df_season = df[df['season'] == season]
    if df_season.empty:
        return None
    
    exact_match = df_season[df_season['name'].astype(str).str.lower() == player_name_lower]
    if not exact_match.empty:
        return exact_match.iloc[0]
    if player_name_norm:
        norm_series = df_season['name'].astype(str).map(normalize_player_name)
        norm_match = df_season[norm_series == player_name_norm]
        if not norm_match.empty:
            return norm_match.iloc[0]
    
    all_seasons_match = df[df['name'].astype(str).str.lower() == player_name_lower]
    if not all_seasons_match.empty:
        candidate_player_ids = all_seasons_match['playerId'].unique()
        for player_id in candidate_player_ids:
            player_in_season = df_season[df_season['playerId'] == player_id]
            if not player_in_season.empty:
                return player_in_season.iloc[0]
    if player_name_norm:
        all_norm_series = df['name'].astype(str).map(normalize_player_name)
        all_norm_match = df[all_norm_series == player_name_norm]
        if not all_norm_match.empty:
            candidate_player_ids = all_norm_match['playerId'].unique()
            for player_id in candidate_player_ids:
                player_in_season = df_season[df_season['playerId'] == player_id]
                if not player_in_season.empty:
                    return player_in_season.iloc[0]
    
    return None

def ensure_player_names(df):
    if df is None or 'name' not in df.columns:
        return df
    name_series = df['name']
    if name_series.notna().mean() >= 0.8:
        return df
    try:
        loader = DataLoader()
        bios = loader.load_player_bios()
        if bios is None or bios.empty or 'name' not in bios.columns:
            return df
        bios = bios[['playerId', 'name']].dropna(subset=['playerId'])
        merged = df.merge(bios, on='playerId', how='left', suffixes=('', '_bio'))
        if 'name_bio' in merged.columns:
            merged['name'] = merged['name'].fillna(merged['name_bio'])
            merged = merged.drop(columns=['name_bio'])
        return merged
    except Exception:
        return df

def initialize_data(force_reload=False):
    if cache['loaded'] and not force_reload:
        return
    
    cache['loaded'] = False
    cache['forwards'] = None
    cache['defensemen'] = None
    cache['forwards_similarity'] = None
    cache['defensemen_similarity'] = None
    forwards_hosted, defensemen_hosted = data_host.load_processed_data()
    fwd_sim_hosted, def_sim_hosted = data_host.load_similarity_data()
    if forwards_hosted is not None and defensemen_hosted is not None:
        cache['forwards'] = ensure_player_names(forwards_hosted)
        cache['defensemen'] = ensure_player_names(defensemen_hosted)
        if fwd_sim_hosted is not None and def_sim_hosted is not None:
            cache['forwards_similarity'] = fwd_sim_hosted
            cache['defensemen_similarity'] = def_sim_hosted
        cache['loaded'] = True

def load_data_in_background():
    loading_state['in_progress'] = True
    loading_state['error'] = None
    
    try:
        initialize_data(force_reload=True)
        if not cache['loaded']:
            error_msg = 'Failed to load data. Please ensure processed parquet files exist in GCS and GCS_BUCKET/GCS_PREFIX are configured.'
            loading_state['error'] = error_msg
    except Exception as e:
        import traceback
        loading_state['error'] = f'{str(e)}\n{traceback.format_exc()}'
    finally:
        loading_state['in_progress'] = False

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/refresh_cache', methods=['POST'])
def refresh_cache():
    try:
        thread = threading.Thread(target=load_data_in_background, daemon=True)
        thread.start()
        return jsonify({'status': 'loading', 'message': 'Cache refresh started'}), 202
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/init', methods=['GET'])
def init():
    if cache['loaded']:
        source = 'hosted' if cache['forwards'] is not None and hasattr(cache['forwards'], 'shape') else 'cache'
        forwards_count = len(cache['forwards']) if cache['forwards'] is not None else 0
        defensemen_count = len(cache['defensemen']) if cache['defensemen'] is not None else 0
        
        return jsonify({
            'status': 'success', 
            'message': f'Data loaded successfully from {source}',
            'details': {
                'forwards_rows': forwards_count,
                'defensemen_rows': defensemen_count,
            }
        })
    
    if loading_state['in_progress']:
        return jsonify({
            'status': 'loading',
            'message': 'Data is being loaded in the background. Please check again in a few moments.'
        }), 202
    
    if loading_state['error']:
        cache_exists = data_host.check_data_available()
        return jsonify({
            'status': 'error',
            'message': loading_state['error'],
            'details': {
                'cache_exists': cache_exists
            }
        }), 500
    
    thread = threading.Thread(target=load_data_in_background, daemon=True)
    thread.start()
    loading_state['thread'] = thread
    
    return jsonify({
        'status': 'loading',
        'message': 'Data loading started in the background. Please check again in a few moments.'
    }), 202

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        cache_exists = data_host.check_data_available()
        
        return jsonify({
            'cacheExists': cache_exists,
            'dataLoaded': cache['loaded']
        })
    
    try:
        if not cache['loaded']:
            forwards_hosted, defensemen_hosted = data_host.load_processed_data()
            if forwards_hosted is not None and defensemen_hosted is not None:
                cache['forwards'] = ensure_player_names(forwards_hosted)
                cache['defensemen'] = ensure_player_names(defensemen_hosted)
                cache['loaded'] = True
            else:
                return jsonify({
                    'error': 'Data not available. Please ensure processed parquet files are present in GCS and GCS_BUCKET/GCS_PREFIX are configured.'
                }), 503
        
        data = request.json or {}
        player_name = data.get('playerName')
        if player_name is None or (isinstance(player_name, str) and player_name.strip() == ""):
            return jsonify({'error': 'playerName is required'}), 400
        player_name = str(player_name).strip()
        player_name_norm = normalize_player_name(player_name)
        season = int(data['season'])
        age = int(data.get('age', 0))
        position = data['position']
        num_neighbors = data.get('numNeighbors', 9)
        filter_season = data.get('filterSeason')
        
        if position and position.upper() != 'D':
            position = 'F'
        
        df = cache['forwards'] if position == 'F' else cache['defensemen']
        if df is not None and hasattr(df, 'columns') and 'WAR' not in df.columns:
            forwards_hosted, defensemen_hosted = data_host.load_processed_data()
            if forwards_hosted is not None and defensemen_hosted is not None:
                cache['forwards'] = forwards_hosted
                cache['defensemen'] = defensemen_hosted
                df = cache['forwards'] if position == 'F' else cache['defensemen']

        similarity_df = cache['forwards_similarity'] if position == 'F' else cache['defensemen_similarity']
        use_pre_transformed = similarity_df is not None and not similarity_df.empty
        if not use_pre_transformed:
            similarity_df = df
        
        player_row = find_player_in_dataframe(df, player_name, season)
        if player_row is None:
            df_season = df[df['season'] == season]
            if not df_season.empty:
                all_names = df_season['name'].dropna().astype(str).unique()
            else:
                all_names = df['name'].dropna().astype(str).unique()
            suggestions = get_close_matches(player_name, all_names, n=5, cutoff=0.6)
            if player_name_norm:
                norm_to_name = {}
                for name in all_names:
                    key = normalize_player_name(name)
                    if key and key not in norm_to_name:
                        norm_to_name[key] = name
                suggestions_norm = get_close_matches(
                    player_name_norm,
                    list(norm_to_name.keys()),
                    n=5,
                    cutoff=0.6
                )
                if suggestions_norm:
                    suggestions = [norm_to_name[k] for k in suggestions_norm]
            
            return jsonify({
                'error': f'Player {player_name} not found in {season}',
                'suggestions': suggestions
            }), 404
        
        actual_player_name = str(player_row['name'])
        
        similarity_engine = SimilarityEngine()
        similar = similarity_engine.find_similar_players(
            similarity_df, actual_player_name, season,
            num_neighbors=num_neighbors,
            metric='l1',
            filter_season=filter_season,
            normalize_first=not use_pre_transformed,
            use_pca=not use_pre_transformed,
            normalization_method='standard'
        )
        
        OFF_METRICS = [
            "SHOT_TAL",
            "PLAY_DRV",
            "SHOT_FREQ",
            "PASS_FREQ",
            "PP_USAGE",
            "ONICE_IMP",
        ]

        DEF_METRICS = [
            "POS_CTRL",
            "BLK",
            "HIT",
            "TAKE",
            "CH_SUP",
            "GOAL_PREV",
        ]


        df_season = df[df['season'] == season].copy()
        time_col = 'I_F_timeOnIce' if 'I_F_timeOnIce' in df_season.columns else 'timeOnIce'
        if time_col not in df_season.columns:
            df_season[time_col] = 1  # fallback
        df_season[time_col] = df_season[time_col].replace(0, 1)  

        
 
        df_season['SHOT_TAL'] = ((df_season.get('I_F_goals_total', 0) - df_season.get('I_F_xGoals_total', 0)) / df_season[time_col] * 60).fillna(0)
        df_season['PLAY_DRV'] = (df_season.get('I_F_primaryAssists_total', 0) / df_season[time_col] * 60).fillna(0)
        df_season['SHOT_FREQ'] = (df_season.get('I_F_shotsOnGoal', 0) / df_season.get('OnIce_F_shotAttempts', 1)).fillna(0)
        df_season['PASS_FREQ'] = ((df_season.get('I_F_primaryAssists_total', 0) + df_season.get('I_F_secondaryAssists_total', 0)) / df_season.get('OnIce_F_shotAttempts', 1)).fillna(0)
        df_season['PP_USAGE'] = (df_season.get('timeOnIcePP', 0) / (df_season.get('timeOnIcePP', 0) + df_season.get('timeOnIcePK', 0) + df_season.get('timeOnIceEV', 0))).fillna(0)
        df_season['ONICE_IMP'] = (df_season.get('OnIce_F_xGoals', 0) / df_season[time_col] * 60).fillna(0)


        df_season['POS_CTRL'] = df_season.get('onIce_corsiPercentage', 0).fillna(0)
        df_season['BLK'] = (df_season.get('shotsBlockedByPlayer', 0) / df_season[time_col] * 60).fillna(0)
        df_season['HIT'] = (df_season.get('I_F_hits', 0) / df_season[time_col] * 60).fillna(0)
        df_season['TAKE'] = (df_season.get('I_F_takeaways', 0) / df_season[time_col] * 60).fillna(0)
        df_season['CH_SUP'] = (df_season.get('OnIce_A_xGoals', 0) / df_season[time_col] * 60).fillna(0)
        df_season['GOAL_PREV'] = (df_season.get('OnIce_A_goals', 0) / df_season[time_col] * 60).fillna(0)


        player_time = player_row.get(time_col, 1)
        if player_time == 0:
            player_time = 1
        player_SHOT_TAL = ((player_row.get('I_F_goals_total', 0) - player_row.get('I_F_xGoals_total', 0)) / player_time * 60)
        player_PLAY_DRV = (player_row.get('I_F_primaryAssists_total', 0) / player_time * 60)

        shot_attempts = player_row.get('OnIce_F_shotAttempts', 1)
        player_SHOT_FREQ = player_row.get('I_F_shotsOnGoal', 0) / shot_attempts


        player_PASS_FREQ = (player_row.get('I_F_primaryAssists_total', 0) +
                           player_row.get('I_F_secondaryAssists_total', 0)) / shot_attempts

        player_PP_USAGE = (player_row.get('timeOnIcePP', 0) / (player_row.get('timeOnIcePP', 0) + player_row.get('timeOnIcePK', 0) + player_row.get('timeOnIceEV', 0)))
        player_ONICE_IMP = (player_row.get('OnIce_F_xGoals', 0) / player_time * 60)

        player_POS_CTRL = player_row.get('onIce_corsiPercentage', 0)
        player_BLK = (player_row.get('shotsBlockedByPlayer', 0) / player_time * 60)
        player_HIT = (player_row.get('I_F_hits', 0) / player_time * 60)
        player_TAKE = (player_row.get('I_F_takeaways', 0) / player_time * 60)
        player_CH_SUP = (player_row.get('OnIce_A_xGoals', 0) / player_time * 60)
        player_GOAL_PREV = (player_row.get('OnIce_A_goals', 0) / player_time * 60)


        offensive_stats = {}
        for metric in OFF_METRICS:
            player_val = locals()[f'player_{metric}']
            percentile = (df_season[metric] < player_val).sum() / len(df_season) * 100
            offensive_stats[metric] = round(percentile, 1)

        defensive_stats = {}
        for metric in DEF_METRICS:
            player_val = locals()[f'player_{metric}']
            percentile = (df_season[metric] < player_val).sum() / len(df_season) * 100
            defensive_stats[metric] = round(percentile, 1)


        for metric in ['CH_SUP', 'GOAL_PREV']:
            if metric in defensive_stats:
                defensive_stats[metric] = round(100 - defensive_stats[metric], 1)
        
        biometrics = {}
        if 'height' in player_row and pd.notna(player_row['height']):
            height_inches = float(player_row['height'])
            feet = int(height_inches // 12)
            inches = int(height_inches % 12)
            biometrics['height'] = f"{feet}'{inches}\""
        if 'weight' in player_row and pd.notna(player_row['weight']):
            biometrics['weight'] = float(player_row['weight'])
        
        team = None
        if 'team' in player_row and pd.notna(player_row['team']):
            team = str(player_row['team'])
        
        archetypes = determine_archetypes(player_row, df, position)
        
        war_percentile = None
        if 'WAR' in player_row and pd.notna(player_row['WAR']):
            df_season = df[df['season'] == season]
            if len(df_season) > 0 and 'WAR' in df_season.columns:
                war_values = df_season['WAR'].dropna()
                if len(war_values) > 0:
                    player_war = float(player_row['WAR'])
                    war_percentile = round((war_values < player_war).sum() / len(war_values) * 100, 1)
        

        games_played = safe_get(player_row, 'games_played_total', safe_get(player_row, 'games_total', 
                        safe_get(player_row, 'GP_total', safe_get(player_row, 'gamesPlayed', 
                        safe_get(player_row, 'games', safe_get(player_row, 'GP', 0))))))
        
        goals = safe_get(player_row, 'I_F_goals_total', safe_get(player_row, 'I_F_goals', 0))
        primary_assists = safe_get(player_row, 'I_F_primaryAssists_total', safe_get(player_row, 'I_F_primaryAssists', 0))
        secondary_assists = safe_get(player_row, 'I_F_secondaryAssists_total', safe_get(player_row, 'I_F_secondaryAssists', 0))
        points = safe_get(player_row, 'I_F_points_total', safe_get(player_row, 'I_F_points', 0))

        penalty_minutes = safe_get(player_row, 'penalityMinutes_total', safe_get(player_row, 'penalityMinutes', 0))
        

        
        total_assists = points - goals


        if games_played == 0:
            time_pp = safe_get(player_row, 'timeOnIcePP', 0)
            time_pk = safe_get(player_row, 'timeOnIcePK', 0)
            time_ev = safe_get(player_row, 'timeOnIceEV', 0)
            total_icetime_seconds = time_pp + time_pk + time_ev
            if total_icetime_seconds > 0:
                icetime_minutes = total_icetime_seconds / 60.0
                games_played = max(1, int(icetime_minutes / 15.0))
            else:
                games_played = 1
        
        stats = {
            'gamesPlayed': int(games_played),
            'goals': int(round(goals)) if goals > 0 else 0,
            'assists': int(round(total_assists)) if total_assists > 0 else 0,
            'points': int(round(points)) if points > 0 else 0,
            'penaltyMinutes': int(round(penalty_minutes)) if penalty_minutes > 0 else 0
        }
        
        result = {
            'player': {
                'name': actual_player_name,
                'season': season,
                'position': position,
                'age': int(player_row['age']) if 'age' in player_row and pd.notna(player_row['age']) else None,
                'playerId': int(player_row['playerId']),
                'team': team,
                'archetypes': archetypes,
                'warPercentile': war_percentile
            },
            'biometrics': biometrics,
            'stats': stats,
            'percentiles': {
                'offensive': offensive_stats,
                'defensive': defensive_stats
            },
            'similarPlayers': similar
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/teams', methods=['GET'])
def get_teams():
    try:
        year = request.args.get('year')
        if not year:
            return jsonify({'error': 'year parameter is required'}), 400
        
        year = int(year)
        
        if not cache['loaded']:
            forwards_hosted, defensemen_hosted = data_host.load_processed_data()
            if forwards_hosted is not None and defensemen_hosted is not None:
                cache['forwards'] = forwards_hosted
                cache['defensemen'] = defensemen_hosted
                cache['loaded'] = True
            else:
                return jsonify({
                    'error': 'Data not available. Please ensure processed parquet files are present in GCS and GCS_BUCKET/GCS_PREFIX are configured.'
                }), 503
        

        df_forwards = cache['forwards']
        df_defensemen = cache['defensemen']
        
        if df_forwards is None or df_defensemen is None:
            return jsonify({'error': 'Data not loaded'}), 500
        

        df_year = pd.concat([
            df_forwards[df_forwards['season'] == year],
            df_defensemen[df_defensemen['season'] == year]
        ])
        
        if df_year.empty:
            return jsonify({'error': f'No data available for year {year}'}), 404
        

        teams = sorted(df_year['team'].dropna().unique().tolist())
        
        return jsonify({
            'year': year,
            'teams': teams
        })
        
    except ValueError:
        return jsonify({'error': 'year must be an integer'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rosters', methods=['GET'])
def get_rosters():
    try:
        year = request.args.get('year')
        team = request.args.get('team')
        position = request.args.get('position')
        

        if not year or not team or not position:
            return jsonify({
                'error': 'year, team, and position parameters are required'
            }), 400
        
        year = int(year)
        team = team.upper()
        position = position.upper()
        
        if position not in ['F', 'D']:
            return jsonify({
                'error': "position must be 'F' (forwards) or 'D' (defensemen)"
            }), 400
        
        if not cache['loaded']:
            forwards_hosted, defensemen_hosted = data_host.load_processed_data()
            if forwards_hosted is not None and defensemen_hosted is not None:
                cache['forwards'] = forwards_hosted
                cache['defensemen'] = defensemen_hosted
                cache['loaded'] = True
            else:
                return jsonify({
                    'error': 'Data not available. Please ensure processed parquet files are present in GCS and GCS_BUCKET/GCS_PREFIX are configured.'
                }), 503
        

        df = cache['forwards'] if position == 'F' else cache['defensemen']
        
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        

        df_filtered = df[(df['season'] == year) & (df['team'] == team)]
        
        if df_filtered.empty:
            return jsonify({
                'error': f'No players found for year {year}, team {team}, position {position}'
            }), 404
        

        df_team_year = df[(df['season'] == year) & (df['team'] == team)]
        

        players = []

        if 'WAR' in df_team_year.columns:

            df = df_filtered.copy()
            df['WAR'] = df['WAR'].fillna(0)

            total_abs_war = df['WAR'].abs().sum()

            k = 2   
            p = 1.5  

            war_adjusted = (df['WAR'])*2
            x = k * war_adjusted / (total_abs_war + 1e-6)

            if total_abs_war > 0:
                df['winShare'] = (
                    99
                    * np.sign(x)
                    * np.abs(np.tanh(x)) ** p
                ).round(1)

            else:
                df['winShare'] = 0.0

            players = [
                {
                    'name': row['name'],
                    'playerId': int(row['playerId']) if pd.notna(row['playerId']) else None,
                    'winShare': row['winShare']
                }
                for _, row in df.iterrows()
            ]

        else:
            players = [
                {
                    'name': player.get('name'),
                    'playerId': int(player.get('playerId')) if pd.notna(player.get('playerId')) else None,
                    'winShare': None
                }
                for _, player in df_filtered.iterrows()
            ]

        players.sort(key=lambda x: x['winShare'] if x['winShare'] is not None else -np.inf, reverse=True)

        return jsonify({
            'year': year,
            'team': team,
            'position': position,
            'players': players
        })
        
    except ValueError as e:
        return jsonify({'error': 'year must be an integer'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
