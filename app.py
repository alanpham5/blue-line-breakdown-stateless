from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import threading
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

env_path = os.path.join(current_dir, '.env')
load_dotenv(env_path)

from utils.similarity_engine import SimilarityEngine
from utils.cache_manager import CacheManager
from utils.data_host import DataHostManager
from difflib import get_close_matches
import pandas as pd

app = Flask(__name__)
CORS(app)

cache = {'forwards': None, 'defensemen': None, 'loaded': False}
loading_state = {'in_progress': False, 'error': None, 'thread': None}
cache_manager = CacheManager()
data_host = DataHostManager()

def find_player_in_dataframe(df, player_name, season):
    player_name_lower = player_name.lower().strip()
    
    df_season = df[df['season'] == season]
    if df_season.empty:
        return None
    
    exact_match = df_season[df_season['name'].str.lower() == player_name_lower]
    if not exact_match.empty:
        return exact_match.iloc[0]
    
    all_seasons_match = df[df['name'].str.lower() == player_name_lower]
    if not all_seasons_match.empty:
        candidate_player_ids = all_seasons_match['playerId'].unique()
        for player_id in candidate_player_ids:
            player_in_season = df_season[df_season['playerId'] == player_id]
            if not player_in_season.empty:
                return player_in_season.iloc[0]
    
    return None

def initialize_data(force_reload=False):
    if cache['loaded'] and not force_reload:
        return
    
    cache['loaded'] = False
    cache['forwards'] = None
    cache['defensemen'] = None
    
    forwards_cached, defensemen_cached = cache_manager.load_processed_data()
    
    if forwards_cached is not None and defensemen_cached is not None:
        cache['forwards'] = forwards_cached
        cache['defensemen'] = defensemen_cached
        cache['loaded'] = True
        return
    
    try:
        forwards_hosted, defensemen_hosted = data_host.load_processed_data()
        if forwards_hosted is not None and defensemen_hosted is not None:
            cache['forwards'] = forwards_hosted
            cache['defensemen'] = defensemen_hosted
            cache['loaded'] = True
            return
    except Exception as e:
        pass

def load_data_in_background():
    loading_state['in_progress'] = True
    loading_state['error'] = None
    
    try:
        initialize_data(force_reload=True)
        if not cache['loaded']:
            cache_exists = cache_manager.cache_exists("forwards_processed.csv") and cache_manager.cache_exists("defensemen_processed.csv")
            github_repo = os.environ.get('GITHUB_REPO', 'Not set')
            
            if github_repo and github_repo != 'Not set':
                base_url = data_host.get_base_url()
                forwards_url = f"{base_url}/forwards_processed.csv" if base_url else "N/A"
                error_msg = f'Failed to load data. No local cache found. GitHub URL returned 404: {forwards_url}. Please create a release tagged "latest" with forwards_processed.csv and defensemen_processed.csv files, or run the data processing script locally to create cache files.'
            else:
                error_msg = 'Failed to load data. No local cache found and GITHUB_REPO not configured. Please set GITHUB_REPO or run the data processing script to create local cache files.'
            
            loading_state['error'] = error_msg
    except Exception as e:
        import traceback
        loading_state['error'] = f'{str(e)}\n{traceback.format_exc()}'
    finally:
        loading_state['in_progress'] = False

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

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
        cache_exists = cache_manager.cache_exists("forwards_processed.csv") and cache_manager.cache_exists("defensemen_processed.csv")
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
        cache_exists = cache_manager.cache_exists("forwards_processed.csv") and \
                     cache_manager.cache_exists("defensemen_processed.csv")
        
        return jsonify({
            'cacheExists': cache_exists,
            'dataLoaded': cache['loaded']
        })
    
    try:
        if not cache['loaded']:
            try:
                forwards_hosted, defensemen_hosted = data_host.load_processed_data()
                if forwards_hosted is not None and defensemen_hosted is not None:
                    cache['forwards'] = forwards_hosted
                    cache['defensemen'] = defensemen_hosted
                    cache['loaded'] = True
                else:
                    raise ValueError("Hosted data not available")
            except Exception:
                forwards_cached, defensemen_cached = cache_manager.load_processed_data()
                if forwards_cached is not None and defensemen_cached is not None:
                    cache['forwards'] = forwards_cached
                    cache['defensemen'] = defensemen_cached
                    cache['loaded'] = True
                else:
                    return jsonify({
                        'error': 'Data not available. Please ensure data files are hosted and DATA_HOST_URL is configured, or run the data processing script to generate local cache files.'
                    }), 503
        
        data = request.json
        player_name = data['playerName']
        season = int(data['season'])
        position = data['position']
        num_neighbors = data.get('numNeighbors', 9)
        filter_season = data.get('filterSeason')
        
        if position and position.upper() != 'D':
            position = 'F'
        
        df = cache['forwards'] if position == 'F' else cache['defensemen']
        
        player_row = find_player_in_dataframe(df, player_name, season)
        if player_row is None:
            df_season = df[df['season'] == season]
            all_names = df_season['name'].unique() if not df_season.empty else df['name'].unique()
            suggestions = get_close_matches(player_name, all_names, n=5, cutoff=0.6)
            
            return jsonify({
                'error': f'Player {player_name} not found in {season}',
                'suggestions': suggestions
            }), 404
        
        actual_player_name = player_row['name']
        
        similarity_engine = SimilarityEngine()
        df_normalized = similarity_engine.normalize_columns(df, method='minmax')
        
        feature_columns_pre = [col for col in df_normalized.columns 
                              if col not in ["playerId", "name", "position", "season", "team"]]
        
        feature_weights_dict = {}
        for col in feature_columns_pre:
            col_str = str(col).lower()
            if 'i_f_' in col_str or 'onice_f_' in col_str:
                if 'goals' in col_str or 'assists' in col_str or 'points' in col_str or 'xgoals' in col_str:
                    feature_weights_dict[col] = 1.3
                else:
                    feature_weights_dict[col] = 1.1
            elif 'hits' in col_str or 'blocked' in col_str or 'takeaways' in col_str:
                feature_weights_dict[col] = 1.2
            elif 'onice_a_' in col_str or 'corsi' in col_str:
                feature_weights_dict[col] = 1.1
            elif 'height' in col_str or 'weight' in col_str or 'bmi' in col_str:
                feature_weights_dict[col] = 1.0
            else:
                feature_weights_dict[col] = 1.0
        
        df_transformed = similarity_engine.pca_transform(df_normalized, n_components=35, feature_weights=feature_weights_dict)
        percentiles = similarity_engine.calculate_percentiles(player_row, df)
        
        similar = similarity_engine.find_similar_players(
            df_transformed, actual_player_name, season, 
            num_neighbors=num_neighbors,
            metric='cosine',
            filter_season=filter_season,
            normalize_first=False,
            use_pca=False
        )
        
        allowed_offensive_stats = [
            'I_F_xGoals', 'I_F_goals', 'I_F_primaryAssists', 'I_F_secondaryAssists',
            'I_F_shotsOnGoal', 'I_F_shotAttempts', 'I_F_points', 'I_F_giveaways',
            'OnIce_F_xGoals', 'OnIce_F_goals'
        ]
        
        allowed_defensive_stats = [
            'OnIce_A_xGoals', 'OnIce_A_goals', 'onIce_corsiPercentage',
            'I_F_hits', 'I_F_takeaways', 'shotsBlockedByPlayer'
        ]
        
        offensive_stats = {}
        for k, v in percentiles.items():
            if k in allowed_offensive_stats:
                if 'giveaways' in k.lower():
                    offensive_stats[k] = round(100 - v, 1)
                else:
                    offensive_stats[k] = v
        
        defensive_stats = {}
        for k, v in percentiles.items():
            if k in allowed_defensive_stats:
                if 'OnIce_A_goals' in k or 'OnIce_A_xGoals' in k:
                    defensive_stats[k] = round(100 - v, 1)
                else:
                    defensive_stats[k] = v
        
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
        
        result = {
            'player': {
                'name': actual_player_name,
                'season': season,
                'position': position,
                'playerId': int(player_row['playerId']),
                'team': team
            },
            'biometrics': biometrics,
            'percentiles': {
                'offensive': offensive_stats,
                'defensive': defensive_stats
            },
            'similarPlayers': similar
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
