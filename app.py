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

def initialize_data(force_reload=False):
    if cache['loaded'] and not force_reload:
        return
    
    forwards_hosted, defensemen_hosted = data_host.load_processed_data()
    
    if forwards_hosted is not None and defensemen_hosted is not None:
        cache['forwards'] = forwards_hosted
        cache['defensemen'] = defensemen_hosted
        cache['loaded'] = True
        return
    
    forwards_cached, defensemen_cached = cache_manager.load_processed_data()
    
    if forwards_cached is not None and defensemen_cached is not None:
        cache['forwards'] = forwards_cached
        cache['defensemen'] = defensemen_cached
        cache['loaded'] = True
        return
    
    raise Exception("Data not available. Please ensure data files are hosted and DATA_HOST_URL is set correctly.")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@app.route('/init', methods=['GET'])
def init():
    try:
        initialize_data(force_reload=True)
        if cache['loaded']:
            return jsonify({
                'status': 'success', 
                'message': 'Data loaded successfully from hosted source'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to load data. Check DATA_HOST_URL configuration.'
            }), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
            forwards_hosted, defensemen_hosted = data_host.load_processed_data()
            if forwards_hosted is not None and defensemen_hosted is not None:
                cache['forwards'] = forwards_hosted
                cache['defensemen'] = defensemen_hosted
                cache['loaded'] = True
            else:
                forwards_cached, defensemen_cached = cache_manager.load_processed_data()
                if forwards_cached is not None and defensemen_cached is not None:
                    cache['forwards'] = forwards_cached
                    cache['defensemen'] = defensemen_cached
                    cache['loaded'] = True
                else:
                    return jsonify({
                        'error': 'Data not available. Please ensure data files are hosted and DATA_HOST_URL is configured.'
                    }), 503
        
        data = request.json
        player_name = data['playerName']
        season = int(data['season'])
        position = data['position']
        num_neighbors = data.get('numNeighbors', 7)
        filter_season = data.get('filterSeason')
        
        if position and position.upper() != 'D':
            position = 'F'
        
        df = cache['forwards'] if position == 'F' else cache['defensemen']
        
        player_data = df[(df['name'] == player_name) & (df['season'] == season)]
        if player_data.empty:
            all_names = df['name'].unique()
            suggestions = get_close_matches(player_name, all_names, n=5, cutoff=0.6)
            
            return jsonify({
                'error': f'Player {player_name} not found in {season}',
                'suggestions': suggestions
            }), 404
        
        similarity_engine = SimilarityEngine()
        df_normalized = similarity_engine.normalize_columns(df, method='minmax')
        df_transformed = similarity_engine.pca_transform(df_normalized, n_components=35)
        
        player_row = player_data.iloc[0]
        percentiles = similarity_engine.calculate_percentiles(player_row, df)
        
        similar = similarity_engine.find_similar_players(
            df_transformed, player_name, season, 
            num_neighbors=num_neighbors,
            metric='cosine',
            filter_season=filter_season
        )
        
        offensive_stats = {k: v for k, v in percentiles.items() 
                          if 'I_F' in k or 'OnIce_F' in k}
        defensive_stats = {k: v for k, v in percentiles.items() 
                          if 'OnIce_A' in k or 'hits' in k or 'takeaways' in k}
        
        result = {
            'player': {
                'name': player_name,
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
