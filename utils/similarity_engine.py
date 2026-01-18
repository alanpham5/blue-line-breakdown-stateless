import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

class SimilarityEngine:
    def __init__(self):
        self.normalizers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
    
    def normalize_columns(self, df, method='minmax'):
        """Normalize columns - should be done BEFORE PCA to match notebook"""
        exclude_cols = ["playerId", "name", "position", "season"]
        df_normalized = df.copy()
        cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
        
        for col in cols_to_normalize:
            if pd.api.types.is_numeric_dtype(df_normalized[col]):
                if method == 'minmax':
                    scaler = MinMaxScaler()
                elif method == 'standard':
                    scaler = StandardScaler()
                elif method == 'robust':
                    scaler = RobustScaler()
                else:
                    scaler = MinMaxScaler()
                df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
        return df_normalized
    
    def pca_transform(self, df, n_components=35):
        """Apply PCA transformation - matches notebook implementation"""
        nonnum_columns = ["playerId", "name", "position", "season"]
        numeric_columns = [col for col in df.columns if col not in nonnum_columns]
        
        df_nonnum = df[nonnum_columns]
        df_numeric = df[numeric_columns]
        
        n_comp = min(n_components, len(numeric_columns))
        pca = PCA(n_components=n_comp)
        trans_df = pca.fit_transform(df_numeric)
        
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
                            metric='euclidean', filter_season=None, 
                            use_pca=True, normalize_first=True, normalization_method='minmax'):
        """
        Find similar players matching the notebook's methodology.
        
        Parameters:
        - df: DataFrame with ALL seasons concatenated
        - player_name: Name of reference player
        - season: Season of reference player
        - num_neighbors: Number of similar players to return
        - metric: Distance metric ('euclidean', 'manhattan', 'cosine', 'chebyshev')
        - filter_season: Filter results to specific season(s) (e.g., "2022-26" or "2022")
        - use_pca: Whether to apply PCA transformation
        - normalize_first: Whether to normalize before PCA
        - normalization_method: 'minmax', 'standard', or 'robust'
        """
        player_df = df[df['season'] == season]
        
        if player_name not in player_df['name'].values:
            raise ValueError(f"Player {player_name} not found in season {season}")
        
        if normalize_first:
            df_processed = self.normalize_columns(df, method=normalization_method)
        else:
            df_processed = df.copy()
        
        if use_pca:
            df_processed = self.pca_transform(df_processed, n_components=35)
        
        player_df_full = df_processed[df_processed['season'] == season]
        if player_name not in player_df_full['name'].values:
            raise ValueError(f"Player {player_name} not found in season {season}")
        
        player_full = player_df_full[player_df_full['name'] == player_name]
        if player_full.empty:
            raise ValueError(f"Player {player_name} not found in season {season}")
        
        player_row = player_full.iloc[0]
        
        feature_columns = [col for col in df_processed.columns 
                          if col not in ["playerId", "name", "position", "season"]]
        player_features_full = player_row[feature_columns].values
        
        if len(player_features_full.shape) > 1:
            player_features_full = player_features_full.flatten()
        
        if filter_season:
            try:
                if '-' in str(filter_season):
                    start, end = map(int, str(filter_season).split('-'))
                    if end < 100:
                        end += (start // 100) * 100
                    df_processed = df_processed[(df_processed['season'] >= start) & 
                                               (df_processed['season'] <= end)]
                else:
                    single_season = int(filter_season)
                    df_processed = df_processed[df_processed['season'] == single_season]
            except ValueError:
                raise ValueError("Invalid filter_season format")
        
        features = df_processed.drop(columns=["playerId", "name", "position", "season"], 
                                     errors='ignore').values
        names = df_processed[['name', 'season', 'playerId', 'position']].values
        
        if len(df_processed) == 0:
            raise ValueError(f"No players found for filter_season: {filter_season}")
        
        model = NearestNeighbors(metric=metric)
        model.fit(features)
        
        distances, indices = model.kneighbors(
            [player_features_full], 
            n_neighbors=min(len(df_processed), num_neighbors + 10)
        )
        
        if len(distances[0]) == 0:
            return []
        
        min_dist, max_dist = np.min(distances[0]), np.max(distances[0])
        if max_dist == min_dist:
            similarities = np.full(len(distances[0]), 100.0)
        else:
            normalized_distances = (distances[0] - min_dist) / (max_dist - min_dist)
            similarities = (1 - normalized_distances) * 100
        
        all_neighbors = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(names):
                all_neighbors.append({
                    'name': str(names[idx][0]),
                    'season': int(names[idx][1]),
                    'playerId': int(names[idx][2]),
                    'position': str(names[idx][3]) if len(names[idx]) > 3 else 'F',
                    'similarity': round(similarity, 1)
                })
        
        neighbors = [n for n in all_neighbors if n['name'] != player_name][:num_neighbors]
        return neighbors