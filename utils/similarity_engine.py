import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

def calculate_feature_weights(feature_columns):
    feature_weights_dict = {}
    for col in feature_columns:
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
    return feature_weights_dict

class SimilarityEngine:
    def __init__(self):
        self.normalizers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
    
    def normalize_columns(self, df, method='minmax'):
        exclude_cols = {"playerId", "name", "position", "season", "team"}
        war_columns = {
            'EV_min', 'PP_min', 'PK_min', 'Off_GAR', 'Def_GAR', 
            'PP_GAR', 'PK_GAR', 'Penalty_GAR', 'Faceoff_GAR', 
            'Total_GAR', 'WAR'
        }
        exclude_cols.update(war_columns)
        df_normalized = df.copy()
        cols_to_normalize = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        if not cols_to_normalize:
            return df_normalized
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()
        
        df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])
        return df_normalized
    
    def pca_transform(self, df, n_components=35, feature_weights=None):
        nonnum_columns = ["playerId", "name", "position", "season", "team"]
        war_columns = {
            'EV_min', 'PP_min', 'PK_min', 'Off_GAR', 'Def_GAR', 
            'PP_GAR', 'PK_GAR', 'Penalty_GAR', 'Faceoff_GAR', 
            'Total_GAR', 'WAR'
        }
        exclude_cols = set(nonnum_columns) | war_columns
        numeric_columns = [col for col in df.columns 
                          if col not in exclude_cols 
                          and pd.api.types.is_numeric_dtype(df[col])]
        
        df_nonnum = df[nonnum_columns]
        df_numeric = df[numeric_columns].copy()
        
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=-1e6, upper=1e6)
        
        if feature_weights is not None:
            weights_array = np.array([feature_weights.get(col, 1.0) for col in numeric_columns])
            df_numeric = (df_numeric * weights_array).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=-1e6, upper=1e6)
        
        variance_threshold = 1e-8
        variances = df_numeric.var()
        valid_columns = variances[variances > variance_threshold].index.tolist()
        df_numeric = df_numeric[valid_columns]
        
        if len(valid_columns) == 0:
            raise ValueError("No valid numeric columns with sufficient variance for PCA")
        
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = StandardScaler()
            df_numeric_scaled = scaler.fit_transform(df_numeric)
            df_numeric = pd.DataFrame(df_numeric_scaled, columns=df_numeric.columns, index=df_numeric.index)
        
        df_numeric_values = np.nan_to_num(
            df_numeric.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=-100, upper=100).values.astype(np.float64),
            nan=0.0, posinf=100.0, neginf=-100.0
        )
        df_numeric_values = np.clip(df_numeric_values, -100, 100)
        
        n_comp = min(n_components, len(valid_columns))
        if n_comp < 1:
            raise ValueError("Not enough components for PCA")
        
        try:
            with np.errstate(all='ignore'):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pca = PCA(n_components=n_comp, svd_solver='auto')
                    trans_df = pca.fit_transform(df_numeric_values)
        except (ValueError, np.linalg.LinAlgError):
            with np.errstate(all='ignore'):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df_numeric_values = np.clip(df_numeric_values, -50, 50)
                    df_numeric_values = np.nan_to_num(df_numeric_values, nan=0.0, posinf=50.0, neginf=-50.0)
                    pca = PCA(n_components=n_comp, svd_solver='arpack')
                    trans_df = pca.fit_transform(df_numeric_values)
        
        new_df = pd.DataFrame(trans_df, index=df.index)
        new_df = pd.concat([df_nonnum.reset_index(drop=True), 
                           new_df.reset_index(drop=True)], axis=1)
        return new_df
    
    def calculate_percentiles(self, player_row, df):
        percentiles = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['playerId', 'season']:
                percentile = (df[col] < player_row[col]).sum() / len(df) * 100
                percentiles[col] = round(percentile, 1)
        
        return percentiles
    
    def find_similar_players(self, df, player_name, season, num_neighbors=7, 
                            metric='cosine', filter_season=None, 
                            use_pca=True, normalize_first=True, normalization_method='minmax'):
        player_df = df[df['season'] == season]
        
        if player_name not in player_df['name'].values:
            raise ValueError(f"Player {player_name} not found in season {season}")
        
        if normalize_first:
            df_processed = self.normalize_columns(df, method=normalization_method)
        else:
            df_processed = df.copy()
        
        if use_pca:
            feature_columns_pre = [col for col in df_processed.columns 
                                  if col not in ["playerId", "name", "position", "season", "team"]]
            feature_weights_dict = calculate_feature_weights(feature_columns_pre)
            df_processed = self.pca_transform(df_processed, n_components=35, feature_weights=feature_weights_dict)
        
        player_df_full = df_processed[df_processed['season'] == season]
        if player_name not in player_df_full['name'].values:
            raise ValueError(f"Player {player_name} not found in season {season}")
        
        player_full = player_df_full[player_df_full['name'] == player_name]
        if player_full.empty:
            raise ValueError(f"Player {player_name} not found in season {season}")
        
        player_row = player_full.iloc[0]
        player_id = int(player_row['playerId'])
        
        if filter_season:
            if '-' in str(filter_season):
                start, end = map(int, str(filter_season).split('-'))
                if end < 100:
                    end += (start // 100) * 100
                df_processed = df_processed[(df_processed['season'] >= start) & 
                                           (df_processed['season'] <= end)]
            else:
                single_season = int(filter_season)
                df_processed = df_processed[df_processed['season'] == single_season]
        
        feature_columns = [col for col in df_processed.columns 
                          if col not in ["playerId", "name", "position", "season", "team"]]
        
        player_features_full = player_row[feature_columns].values
        
        if len(player_features_full.shape) > 1:
            player_features_full = player_features_full.flatten()
        
        features = df_processed.drop(columns=["playerId", "name", "position", "season", "team"], 
                                     errors='ignore').values
        
        if 'team' in df_processed.columns:
            names = df_processed[['name', 'season', 'playerId', 'position', 'team']].values
        else:
            names = df_processed[['name', 'season', 'playerId', 'position']].values
        
        if len(df_processed) == 0:
            raise ValueError(f"No players found for filter_season: {filter_season}")
        
        model = NearestNeighbors(metric=metric, n_jobs=-1)
        model.fit(features)
        
        sample_size = min(len(df_processed), 3000)
        sample_indices = np.random.choice(len(df_processed), size=sample_size, replace=False) if len(df_processed) > sample_size else np.arange(len(df_processed))
        
        sample_distances, _ = model.kneighbors(
            [player_features_full],
            n_neighbors=min(sample_size, len(df_processed))
        )
        
        if len(sample_distances[0]) == 0:
            return []
        
        reference_distances = sample_distances[0]
        reference_min = np.min(reference_distances)
        reference_p10 = np.percentile(reference_distances, 10)
        reference_p25 = np.percentile(reference_distances, 25)
        reference_p50 = np.percentile(reference_distances, 50)
        reference_p75 = np.percentile(reference_distances, 75)
        reference_p90 = np.percentile(reference_distances, 90)
        
        iqr = reference_p75 - reference_p25
        if iqr == 0:
            iqr = max(reference_p50 - reference_min, reference_p90 - reference_p10, 0.01)
        
        scale_factor = iqr * 2.0
        
        requested_neighbors = min(len(df_processed), num_neighbors + 20)
        
        distances, indices = model.kneighbors(
            [player_features_full], 
            n_neighbors=requested_neighbors
        )
        
        if len(distances[0]) == 0:
            return []
        
        normalized_distances = (distances[0] - reference_min) / scale_factor
        similarities = 100 * np.exp(-normalized_distances)
        
        max_similarity = np.max(similarities)
        min_similarity = np.min(similarities)
        similarity_range = max_similarity - min_similarity
        
        if similarity_range > 0:
            scaled_similarities = 30 + (similarities - min_similarity) / similarity_range * 65
            similarities = scaled_similarities
        
        similarities = np.clip(similarities, 0, 95)
        
        all_neighbors = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(names):
                neighbor_data = {
                    'name': str(names[idx][0]),
                    'season': int(names[idx][1]),
                    'playerId': int(names[idx][2]),
                    'position': str(names[idx][3]) if len(names[idx]) > 3 else 'F',
                    'similarity': float(similarity)
                }
                if len(names[idx]) > 4 and pd.notna(names[idx][4]):
                    neighbor_data['team'] = str(names[idx][4])
                all_neighbors.append(neighbor_data)
        
        candidate_neighbors = [n for n in all_neighbors if n['playerId'] != player_id]
        
        player_self = {
            'name': player_name,
            'season': season,
            'playerId': player_id,
            'position': str(player_row['position']) if 'position' in player_row else 'F',
            'similarity': 99.0
        }
        if 'team' in player_row and pd.notna(player_row['team']):
            player_self['team'] = str(player_row['team'])
        candidate_neighbors.append(player_self)
        
        for n in candidate_neighbors:
            if n['playerId'] != player_id:
                season_diff = abs(n['season'] - season)
                era_penalty = 30 - season_diff
                n['similarity'] = n['similarity'] - era_penalty
        
        if len(candidate_neighbors) > 0:
            all_similarities = [n['similarity'] for n in candidate_neighbors]
            max_sim = max(all_similarities)
            min_sim = min(all_similarities)
            sim_range = max_sim - min_sim
            
            if sim_range > 0:
                target_max = 99.0
                target_min = 50.0
                scale_factor = (target_max - target_min) / sim_range
                for n in candidate_neighbors:
                    n['similarity'] = round(target_min + (n['similarity'] - min_sim) * scale_factor, 1)
        
        candidate_neighbors = [n for n in candidate_neighbors if n['playerId'] != player_id]
        candidate_neighbors.sort(key=lambda x: x['similarity'], reverse=True)
        
        neighbors = []
        season_counts = {}
        max_per_season = max(2, num_neighbors // 3)
        
        for n in candidate_neighbors:
            n_season = n['season']
            current_count = season_counts.get(n_season, 0)
            
            if current_count < max_per_season or len(neighbors) < num_neighbors:
                neighbors.append(n)
                season_counts[n_season] = current_count + 1
                if len(neighbors) >= num_neighbors:
                    break
        
        if len(neighbors) < num_neighbors:
            neighbors_set = {n['playerId'] for n in neighbors}
            for n in candidate_neighbors:
                if n['playerId'] not in neighbors_set:
                    neighbors.append(n)
                    neighbors_set.add(n['playerId'])
                    if len(neighbors) >= num_neighbors:
                        break
        
        neighbors.sort(key=lambda x: x['similarity'], reverse=True)
        return neighbors