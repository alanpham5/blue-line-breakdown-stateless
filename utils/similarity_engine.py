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
                w = 1.8

        elif 'onice_f_' in c:
            if 'xgoals' in c:
                w = 1.4
            elif 'shots' in c:
                w = 1.3
            else:
                w = 1.0

        elif 'onice_a_' in c:
            w = 0.8

        elif 'corsi' in c:
            w = 0.7

        elif any(x in c for x in ['hits', 'I_F_hits', 'penalityMinutes','penaltyMinutes' , 'penaltiesTakenEV']):
            w = 2.4

        elif any(x in c for x in ['takeaways', 'blocked']):
            w = 2

        elif c in ['war', 'off_gar', 'def_gar', 'pp_gar', 'pk_gar', 'gamescore']:
            w = 0.9

        elif c in ['height', 'weight']:
            w = 0.8
        
        elif c == 'age':
            w = 1.3

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

        df_numeric_values = np.nan_to_num(
            df_numeric_scaled.astype(np.float64),
            nan=0.0, posinf=100.0, neginf=-100.0
        )
        df_numeric_values = np.clip(df_numeric_values, -100, 100)

        n_comp = min(n_components, df_numeric_values.shape[1])
        if n_comp < 1:
            raise ValueError("Not enough components for PCA")

        try:
            pca = PCA(n_components=n_comp)
            trans_df = pca.fit_transform(df_numeric_values)
        except (ValueError, np.linalg.LinAlgError):
            df_numeric_values = np.clip(df_numeric_values, -50, 50)
            pca = PCA(n_components=n_comp, svd_solver='arpack')
            trans_df = pca.fit_transform(df_numeric_values)

        new_df = pd.DataFrame(trans_df, index=df.index)
        new_df = pd.concat([df_nonnum.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)
        return new_df

    def distance_to_similarity(self, distances, metric):
        distances = np.asarray(distances)
        if metric in ["cosine", "correlation"]:
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

    def find_similar_players(
        self,
        df,
        player_name,
        season,
        num_neighbors=7,
        metric='cosine',
        filter_season=None,
        use_pca=True,
        normalize_first=True,
        normalization_method='minmax'
    ):
        df_processed = df.copy()

        player_mask = (df_processed['name'] == player_name) & (df_processed['season'] == season)
        if not player_mask.any():
            raise ValueError(f"Player {player_name} not found in season {season}")

        player_row_raw = df_processed[player_mask].iloc[0]

        minute_cols = ['timeOnIceEV', 'timeOnIcePP', 'timeOnIcePK']


        if minute_cols is not None:
            df_processed = df_processed.copy()
            df_processed['_total_min'] = (
                df_processed[minute_cols[0]] +
                df_processed[minute_cols[1]] +
                df_processed[minute_cols[2]]
            )

            season_thresholds = df_processed.groupby('season')['_total_min'].transform(
                lambda x: x.quantile(0.4)
            )

            filtered_df = df_processed[df_processed['_total_min'] >= season_thresholds]

            player_total_min = (
                player_row_raw[minute_cols[0]] +
                player_row_raw[minute_cols[1]] +
                player_row_raw[minute_cols[2]]
            )

            # Get the threshold for the input player's season
            player_threshold = season_thresholds[player_mask].iloc[0]

            # Ensure the input player is always present, even if they don't meet the threshold
            if player_total_min < player_threshold:
                filtered_df = pd.concat(
                    [filtered_df, df_processed[player_mask]],
                    axis=0
                )

            df_processed = filtered_df.drop(columns=['_total_min'])

        if normalize_first:
            df_processed = self.normalize_columns(df_processed, method=normalization_method)

        if use_pca:
            feature_columns_pre = [
                col for col in df_processed.columns
                if col not in ["playerId", "name", "position", "season", "team"]
            ]
            feature_weights_dict = calculate_feature_weights(feature_columns_pre)
            df_processed = self.pca_transform(
                df_processed,
                n_components=35,
                feature_weights=feature_weights_dict
            )

        feature_columns = [
            col for col in df_processed.columns
            if col not in ["playerId", "name", "position", "season", "team"]
        ]

        player_row = df_processed[
            (df_processed['name'] == player_name) &
            (df_processed['season'] == season)
        ].iloc[0]

        player_features = player_row[feature_columns].values
        player_id = int(player_row['playerId'])

        features = df_processed[feature_columns].values

        names = df_processed[['name', 'season', 'playerId', 'position', 'team']].values

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
                'position': str(names[idx][3]),
                'similarity': float(sim)
            }
            if pd.notna(names[idx][4]):
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
