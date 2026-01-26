import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

exclude_cols = {
    "playerId", "name", "season", "team",
    "EV_min", "PP_min", "PK_min",
    "timeOnIceEV", "timeOnIcePP", "timeOnIcePK",
    "Total_GAR"
}

def calculate_feature_weights(feature_columns):
    feature_weights_dict = {}
    for col in feature_columns:
        c = str(col).lower()
        w = 1.0

 
        if 'i_f_' in c:
            if 'goals' in c or 'xgoals' in c:
                w = 2.2 
            elif 'primaryassists' in c:
                w = 2
            elif 'secondaryassists' in c:
                w = 1.5
            elif 'shots' in c or 'shotattempts' in c:
                w = 1.5 
            elif 'points' in c:
                w = 0.6

        elif 'onice_f_' in c:
            if 'xgoals' in c:
                w = 1.3
            elif 'shots' in c:
                w = 1.2
            else:
                w = 1.0


        elif 'onice_a_' in c:
            w = 0.8  


        elif 'corsi' in c:
            w = 0.7  

 
        elif any(x in c for x in ['takeaways', 'blocked']):
            w = 2
        elif 'hits' in c:
            w = 2

        elif c in ['war', 'off_gar', 'def_gar', 'pp_gar', 'pk_gar', 'gamescore']:
            w = 0.5

        elif any(x in c for x in ['height', 'weight', 'bmi']):
            w = 0.7 

        elif c == 'age':
            w = 1

        feature_weights_dict[col] = w

    return feature_weights_dict

class SimilarityEngine:
    def __init__(self):
        self.normalizers = {
            'minmax': MinMaxScaler(),
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }

    def normalize_columns(self, df, method='minmax'):
        df_normalized = df.copy()
        numeric_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            return df_normalized
        if method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'standard':
            scaler_class = StandardScaler
        elif method == 'robust':
            scaler_class = RobustScaler
        else:
            scaler_class = MinMaxScaler
        seasons = df_normalized['season'].unique()
        normalized_dfs = []
        for season in seasons:
            season_df = df_normalized[df_normalized['season'] == season].copy()
            scaler = scaler_class()
            season_df[numeric_cols] = scaler.fit_transform(season_df[numeric_cols])
            normalized_dfs.append(season_df)
        df_normalized = pd.concat(normalized_dfs, axis=0).sort_index()
        return df_normalized

    def pca_transform(self, df, n_components=35, feature_weights=None):
        nonnum_columns = ["playerId", "name", "position", "season", "team"]
        numeric_columns = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        df_nonnum = df[nonnum_columns]
        df_numeric = df[numeric_columns].copy()
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=-1e6, upper=1e6)
        scaler = StandardScaler()
        df_numeric_scaled = scaler.fit_transform(df_numeric)
        if feature_weights is not None:
            weights_array = np.array([feature_weights.get(col, 1.0) for col in numeric_columns])
            df_numeric_scaled = df_numeric_scaled * weights_array
        df_numeric = pd.DataFrame(df_numeric_scaled, columns=df_numeric.columns, index=df_numeric.index)
        df_numeric_values = np.nan_to_num(df_numeric.values.astype(np.float64), nan=0.0, posinf=100.0, neginf=-100.0)
        df_numeric_values = np.clip(df_numeric_values, -100, 100)
        n_comp = min(n_components, df_numeric_values.shape[1])
        if n_comp < 1:
            raise ValueError("Not enough components for PCA")
        try:
            pca = PCA(n_components=n_comp, svd_solver='auto')
            trans_df = pca.fit_transform(df_numeric_values)
        except (ValueError, np.linalg.LinAlgError):
            df_numeric_values = np.clip(df_numeric_values, -50, 50)
            df_numeric_values = np.nan_to_num(df_numeric_values, nan=0.0, posinf=50.0, neginf=-50.0)
            pca = PCA(n_components=n_comp, svd_solver='arpack')
            trans_df = pca.fit_transform(df_numeric_values)
        new_df = pd.DataFrame(trans_df, index=df.index)
        new_df = pd.concat([df_nonnum.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)
        return new_df

    def calculate_percentiles(self, player_row, df):
        percentiles = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['playerId', 'season']:
                percentile = (df[col] < player_row[col]).sum() / len(df) * 100
                percentiles[col] = round(percentile, 1)
        return percentiles

    def distance_to_similarity(self, distances, metric):
        distances = np.asarray(distances)
        if metric == "cosine":
            similarities = 1 - distances
            similarities = np.clip(similarities, -1, 1)
            similarities = (similarities + 1) / 2 * 100
        elif metric in ["correlation"]:
            similarities = 1 - distances
            similarities = np.clip(similarities, -1, 1)
            similarities = (similarities + 1) / 2 * 100
        else:
            d_min = distances.min()
            d_max = distances.max()
            if d_max == d_min:
                similarities = np.ones_like(distances) * 100
            else:
                similarities = 100 * (1 - (distances - d_min) / (d_max - d_min))
        return np.clip(similarities, 0, 100)

    def find_similar_players(self, df, player_name, season, num_neighbors=7, metric='cosine', filter_season=None, use_pca=True, normalize_first=True, normalization_method='minmax'):
        df_processed = df.copy()
        if normalize_first:
            df_processed = self.normalize_columns(df_processed, method=normalization_method)
        if use_pca:
            feature_columns_pre = [col for col in df_processed.columns if col not in ["playerId", "name", "position", "season", "team"]]
            feature_weights_dict = calculate_feature_weights(feature_columns_pre)
            df_processed = self.pca_transform(df_processed, n_components=35, feature_weights=feature_weights_dict)
        player_mask = (df_processed['name'] == player_name) & (df_processed['season'] == season)
        if not player_mask.any():
            raise ValueError(f"Player {player_name} not found in season {season}")
        feature_columns = [col for col in df_processed.columns if col not in ["playerId", "name", "position", "season", "team"]]
        player_row = df_processed[player_mask].iloc[0]
        player_features = player_row[feature_columns].values
        player_id = int(player_row['playerId'])
        features = df_processed[feature_columns].values
        if 'team' in df_processed.columns:
            names = df_processed[['name', 'season', 'playerId', 'position', 'team']].values
        else:
            names = df_processed[['name', 'season', 'playerId', 'position']].values
        model = NearestNeighbors(metric=metric, n_jobs=-1)
        model.fit(features)
        distances, indices = model.kneighbors([player_features], n_neighbors=len(df_processed))
        distances = distances.flatten()
        similarities = self.distance_to_similarity(distances, metric)
        all_neighbors = []
        for idx, sim in zip(indices.flatten(), similarities):
            neighbor_data = {
                'name': str(names[idx][0]),
                'season': int(names[idx][1]),
                'playerId': int(names[idx][2]),
                'position': str(names[idx][3]) if len(names[idx]) > 3 else 'F',
                'similarity': float(sim)
            }
            if len(names[idx]) > 4 and pd.notna(names[idx][4]):
                neighbor_data['team'] = str(names[idx][4])
            all_neighbors.append(neighbor_data)
        if filter_season:
            if '-' in str(filter_season):
                start, end = map(int, str(filter_season).split('-'))
                if end < 100:
                    end += (start // 100) * 100
                all_neighbors = [n for n in all_neighbors if start <= n['season'] <= end]
            else:
                single_season = int(filter_season)
                all_neighbors = [n for n in all_neighbors if n['season'] == single_season]
        neighbors = [n for n in all_neighbors if n['playerId'] != player_id]
        neighbors.sort(key=lambda x: x['similarity'], reverse=True)
        return neighbors[:num_neighbors]
