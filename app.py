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

from utils.similarity_engine import SimilarityEngine, calculate_feature_weights
from utils.cache_manager import CacheManager
from utils.data_host import DataHostManager
from difflib import get_close_matches
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
        goals_against_p50 = safe_quantile(df, 'OnIce_A_goals', 0.50, float('inf'))
        blocked_p75 = safe_quantile(df, 'shotsBlockedByPlayer', 0.75)
        penalty_minutes_p50 = safe_quantile(df, 'penaltyMinutes', 0.50)
        
        if goals >= goals_p75 and shot_attempts >= shot_attempts_p75:
            archetypes.append('Sniper')
        
        if total_assists >= assists_p75 and points >= points_p75 and assists_to_goals >= 1.2 and total_assists > goals:
            archetypes.append('Playmaker')
        
        if (hits >= hits_p60 or blocked_shots >= blocked_p75) and points >= points_p60 and (height >= 72 and weight >= 200):
            archetypes.append('Power Forward')
        
        if ((blocked_shots >= blocked_p75) or (takeaways >= takeaways_p75)) and points < points_p60:
            archetypes.append('Defensive Forward')
        
        if points >= points_p75 and goals_against <= goals_against_p50 and (takeaways >= takeaways_p75 or blocked_shots >= blocked_p75):
            archetypes.append('Two-Way')
        
        if hits >= hits_p75 and points < points_p60 and penalty_minutes >= penalty_minutes_p50:
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
        
        if len(df) == 0:
            return archetypes
        
        total_assists = primary_assists + secondary_assists
        
        points_p75 = safe_quantile(df, 'I_F_points', 0.75)
        goals_p75 = safe_quantile(df, 'I_F_goals', 0.75)
        assists_p75 = safe_quantile(df, 'I_F_primaryAssists', 0.75)
        shot_attempts_p75 = safe_quantile(df, 'I_F_shotAttempts', 0.75)
        hits_p75 = safe_quantile(df, 'I_F_hits', 0.75)
        blocked_p75 = safe_quantile(df, 'shotsBlockedByPlayer', 0.75)
        goals_against_p75 = safe_quantile(df, 'OnIce_A_goals', 0.75, float('inf'))
        corsiPercentage_p75 = safe_quantile(df, 'onIce_corsiPercentage', 0.75)
        penalty_minutes_p50 = safe_quantile(df, 'penaltyMinutes', 0.50)
        
        if shot_attempts >= shot_attempts_p75 and goals >= goals_p75:
            archetypes.append('Point Shooter')
        
        if points >= points_p75 and (total_assists >= assists_p75 or goals >= goals_p75):
            archetypes.append('Offensive Puck-Mover')
        
        if points >= points_p75 and goals_against <= goals_against_p75 and corsiPercentage >= corsiPercentage_p75:
            archetypes.append('Two-Way')
        
        if hits >= hits_p75 and points < points_p75 and penalty_minutes >= penalty_minutes_p50:
            archetypes.append('Grinder')
        
        if blocked_shots >= blocked_p75:
            archetypes.append('Shot Blocker')
    
    return archetypes

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
    
    forwards_hosted, defensemen_hosted = data_host.load_processed_data()
    if forwards_hosted is not None and defensemen_hosted is not None:
        cache['forwards'] = forwards_hosted
        cache['defensemen'] = defensemen_hosted
        cache['loaded'] = True

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
            forwards_cached, defensemen_cached = cache_manager.load_processed_data()
            if forwards_cached is not None and defensemen_cached is not None:
                cache['forwards'] = forwards_cached
                cache['defensemen'] = defensemen_cached
                cache['loaded'] = True
            else:
                forwards_hosted, defensemen_hosted = data_host.load_processed_data()
                if forwards_hosted is not None and defensemen_hosted is not None:
                    cache['forwards'] = forwards_hosted
                    cache['defensemen'] = defensemen_hosted
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
        if df is not None and hasattr(df, 'columns') and 'WAR' not in df.columns:
            forwards_cached, defensemen_cached = cache_manager.load_processed_data()
            if forwards_cached is not None and defensemen_cached is not None:
                cache['forwards'] = forwards_cached
                cache['defensemen'] = defensemen_cached
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
        
        war_columns = {
            'EV_min', 'PP_min', 'PK_min', 'Off_GAR', 'Def_GAR', 
            'PP_GAR', 'PK_GAR', 'Penalty_GAR', 'Faceoff_GAR', 
            'Total_GAR', 'WAR'
        }
        feature_columns_pre = [col for col in df_normalized.columns 
                              if col not in ["playerId", "name", "position", "season", "team"] 
                              and col not in war_columns]
        
        feature_weights_dict = calculate_feature_weights(feature_columns_pre)
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
        
        allowed_offensive_stats = {
            'I_F_xGoals', 'I_F_goals', 'I_F_primaryAssists', 'I_F_secondaryAssists',
            'I_F_shotsOnGoal', 'I_F_shotAttempts', 'I_F_points', 'I_F_giveaways',
            'OnIce_F_xGoals', 'OnIce_F_goals'
        }
        
        allowed_defensive_stats = {
            'OnIce_A_xGoals', 'OnIce_A_goals', 'onIce_corsiPercentage',
            'I_F_hits', 'I_F_takeaways', 'shotsBlockedByPlayer'
        }
        
        offensive_stats = {
            k: round(100 - v, 1) if 'giveaways' in k.lower() else v
            for k, v in percentiles.items()
            if k in allowed_offensive_stats
        }
        
        defensive_stats = {
            k: round(100 - v, 1) if 'OnIce_A_goals' in k or 'OnIce_A_xGoals' in k else v
            for k, v in percentiles.items()
            if k in allowed_defensive_stats
        }
        
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
        
        result = {
            'player': {
                'name': actual_player_name,
                'season': season,
                'position': position,
                'playerId': int(player_row['playerId']),
                'team': team,
                'archetypes': archetypes,
                'warPercentile': war_percentile
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

@app.route('/teams', methods=['GET'])
def get_teams():
    try:
        year = request.args.get('year')
        if not year:
            return jsonify({'error': 'year parameter is required'}), 400
        
        year = int(year)
        
        # Load data if not already cached
        if not cache['loaded']:
            forwards_cached, defensemen_cached = cache_manager.load_processed_data()
            if forwards_cached is not None and defensemen_cached is not None:
                cache['forwards'] = forwards_cached
                cache['defensemen'] = defensemen_cached
                cache['loaded'] = True
            else:
                forwards_hosted, defensemen_hosted = data_host.load_processed_data()
                if forwards_hosted is not None and defensemen_hosted is not None:
                    cache['forwards'] = forwards_hosted
                    cache['defensemen'] = defensemen_hosted
                    cache['loaded'] = True
                else:
                    return jsonify({
                        'error': 'Data not available. Please ensure data files are hosted and DATA_HOST_URL is configured, or run the data processing script to generate local cache files.'
                    }), 503
        
        # Combine forwards and defensemen data
        df_forwards = cache['forwards']
        df_defensemen = cache['defensemen']
        
        if df_forwards is None or df_defensemen is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Filter by year
        df_year = pd.concat([
            df_forwards[df_forwards['season'] == year],
            df_defensemen[df_defensemen['season'] == year]
        ])
        
        if df_year.empty:
            return jsonify({'error': f'No data available for year {year}'}), 404
        
        # Get unique team abbreviations and sort them
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
        
        # Validate parameters
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
        
        # Load data if not already cached
        if not cache['loaded']:
            forwards_cached, defensemen_cached = cache_manager.load_processed_data()
            if forwards_cached is not None and defensemen_cached is not None:
                cache['forwards'] = forwards_cached
                cache['defensemen'] = defensemen_cached
                cache['loaded'] = True
            else:
                forwards_hosted, defensemen_hosted = data_host.load_processed_data()
                if forwards_hosted is not None and defensemen_hosted is not None:
                    cache['forwards'] = forwards_hosted
                    cache['defensemen'] = defensemen_hosted
                    cache['loaded'] = True
                else:
                    return jsonify({
                        'error': 'Data not available. Please ensure data files are hosted and DATA_HOST_URL is configured, or run the data processing script to generate local cache files.'
                    }), 503
        
        # Select appropriate dataset based on position
        df = cache['forwards'] if position == 'F' else cache['defensemen']
        
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Filter by year and team
        df_filtered = df[(df['season'] == year) & (df['team'] == team)]
        
        if df_filtered.empty:
            return jsonify({
                'error': f'No players found for year {year}, team {team}, position {position}'
            }), 404
        
        # Filter by team for WAR percentile calculation (all players on team for that year)
        df_team_year = df[(df['season'] == year) & (df['team'] == team)]
        
        # Build player list with WAR percentiles
        players = []
        if 'WAR' in df_team_year.columns:
            war_values = df_team_year['WAR'].dropna()
            
            for _, player in df_filtered.iterrows():
                player_war = player.get('WAR')
                
                # Calculate win share (WAR percentile within the team)
                win_share = None
                if pd.notna(player_war) and len(war_values) > 0:
                    percentile = (war_values < player_war).sum() / len(war_values) * 100
                    win_share = round(percentile, 1)
                
                players.append({
                    'name': player.get('name'),
                    'playerId': int(player.get('playerId')) if pd.notna(player.get('playerId')) else None,
                    'winShare': win_share
                })
        else:
            # If WAR column doesn't exist, return players without winShare
            for _, player in df_filtered.iterrows():
                players.append({
                    'name': player.get('name'),
                    'playerId': int(player.get('playerId')) if pd.notna(player.get('playerId')) else None,
                    'winShare': None
                })
        
        # Sort by winShare descending (highest first)
        players.sort(key=lambda x: x['winShare'] if x['winShare'] is not None else -1, reverse=True)
        
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
