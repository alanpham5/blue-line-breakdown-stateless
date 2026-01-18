from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
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
    
    try:
        forwards_hosted, defensemen_hosted = data_host.load_processed_data()
        if forwards_hosted is not None and defensemen_hosted is not None:
            cache['forwards'] = forwards_hosted
            cache['defensemen'] = defensemen_hosted
            cache['loaded'] = True
            return
    except Exception as e:
        pass
    
    forwards_cached, defensemen_cached = cache_manager.load_processed_data()
    
    if forwards_cached is not None and defensemen_cached is not None:
        cache['forwards'] = forwards_cached
        cache['defensemen'] = defensemen_cached
        cache['loaded'] = True
        return
    

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/init', methods=['GET'])
def init():
    import time
    
    max_wait_time = 120
    retry_interval = 50
    start_time = time.time()
    
    try:
        while time.time() - start_time < max_wait_time:
            initialize_data(force_reload=True)
            
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
            
            time.sleep(retry_interval)
        
        cache_exists = cache_manager.cache_exists("forwards_processed.csv") and cache_manager.cache_exists("defensemen_processed.csv")
        return jsonify({
            'status': 'error',
            'message': f'Failed to load data after {max_wait_time} seconds. Please ensure data files exist.',
            'details': {
                'cache_exists': cache_exists,
                'waited_seconds': max_wait_time
            }
        }), 500
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

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
        df_transformed = similarity_engine.pca_transform(df_normalized, n_components=35)
        percentiles = similarity_engine.calculate_percentiles(player_row, df)
        
        similar = similarity_engine.find_similar_players(
            df_transformed, actual_player_name, season, 
            num_neighbors=num_neighbors,
            metric='cosine',
            filter_season=filter_season
        )
        
        offensive_stats = {k: v for k, v in percentiles.items() 
                          if 'I_F' in k or 'OnIce_F' in k}
        
        defensive_stats = {}
        for k, v in percentiles.items():
            k_lower = k.lower()
            if 'OnIce_A' in k or 'hits' in k_lower or 'takeaways' in k_lower or 'shotsBlockedByPlayer' in k or 'Blocked' in k or 'corsipercentage' in k_lower:
                if 'OnIce_A_goals' in k or 'OnIce_A_xGoals' in k:
                    defensive_stats[k] = round(100 - v, 1)
                else:
                    defensive_stats[k] = v
        
        result = {
            'player': {
                'name': actual_player_name,
                'season': season,
                'position': position,
                'playerId': int(player_row['playerId'])
            },
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
