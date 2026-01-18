import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.impute import KNNImputer

class DataProcessor:
    def __init__(self):
        self.icetime_relative_features = [
            "playerId", "name", "season", 'height', 'weight', "age", "position",
            "I_F_xGoals", "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
            "I_F_shotsOnGoal", "I_F_shotAttempts", "I_F_points", "I_F_hits",
            "I_F_takeaways", "I_F_giveaways", "shotsBlockedByPlayer",
            "OnIce_F_xGoals", "OnIce_F_goals",
            "OnIce_A_xGoals", "OnIce_A_goals", "onIce_corsiPercentage"
        ]
        
        self.all_numeric_features = None
        self.nonnum_columns = ["playerId", "name", "position", "season"]
        
        self.team_abbrev_cleanup = {
            'S.J': 'SJS',
            'S. J': 'SJS',
            'S.J.': 'SJS',
            'L.A': 'LAK',
            'L. A': 'LAK',
            'L.A.': 'LAK',
            'N.J': 'NJD',
            'N. J': 'NJD',
            'N.J.': 'NJD',
            'T.B': 'TBL',
            'T. B': 'TBL',
            'T.B.': 'TBL',
            'N.Y.I': 'NYI',
            'N.Y. I': 'NYI',
            'N.Y.I.': 'NYI',
            'N.Y.R': 'NYR',
            'N.Y. R': 'NYR',
            'N.Y.R.': 'NYR',
            'S.T.L': 'STL',
            'S.T. L': 'STL',
            'S.T.L.': 'STL',
            'T.O': 'TOR',
            'T. O': 'TOR',
            'T.O.': 'TOR',
        }
    
    def convert_height_to_inches(self, height_str):
        if pd.isna(height_str):
            return np.nan
        match = re.match(r"(\d+)['\-]\s*(\d+)", height_str)
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            return feet * 12 + inches
        return np.nan
    
    def calculate_age(self, row):
        season_year = int(row['season'])
        season_date = datetime(season_year, 10, 1)
        age = season_date.year - row['birthDate'].year - (
            (season_date.month, season_date.day) < 
            (row['birthDate'].month, row['birthDate'].day)
        )
        return age
    
    def merge_player_data(self, df_stats, df_info):
        df_info['birthDate'] = pd.to_datetime(df_info['birthDate'])
        merged_df = pd.merge(df_stats, df_info, on='playerId', how='inner')
        merged_df['age'] = merged_df.apply(self.calculate_age, axis=1)
        columns = ['playerId', 'height', 'weight', 'age'] + \
                  [col for col in df_stats.columns if col not in ['playerId']]
        return merged_df[columns]
    
    def add_bmi(self, df):
        df['bmi'] = df['weight'] / (df['height'] ** 2) * 703
        return df
    
    def clean_team_abbreviations(self, df):
        if 'team' not in df.columns:
            return df
        
        df = df.copy()
        df['team'] = df['team'].astype(str).replace(self.team_abbrev_cleanup)
        df['team'] = df['team'].replace('nan', np.nan)
        return df
    
    def scale_stats_per_60_min(self, df):
        if 'icetime' not in df.columns:
            return df
        
        df['icetime_hours'] = df['icetime'] / 3600
        
        exclude_cols = ["playerId", "name", "season", 'height', 'weight', 
                       "age", "bmi", "position", "icetime", "onIce_corsiPercentage",
                       "icetime_hours"]
        
        exclude_cols.append('team')
        
        stats_columns = [col for col in df.columns 
                        if col not in exclude_cols 
                        and pd.api.types.is_numeric_dtype(df[col])]
        
        if not stats_columns:
            return df
        
        df_scaled = df.copy()
        df_scaled[stats_columns] = df[stats_columns].div(df['icetime_hours'], axis=0)
        
        cols_to_drop = ['icetime_hours']
        if 'icetime' in df_scaled.columns:
            cols_to_drop.append('icetime')
        df_scaled = df_scaled.drop(columns=cols_to_drop)
        return df_scaled
    
    def process_data(self, df, player_bio):
        merged_data = self.merge_player_data(df, player_bio)
        all_data = merged_data[merged_data['situation'] == 'all']
        
        if 'icetime' in all_data.columns:
            all_data = all_data[all_data['icetime'] >= all_data['icetime'].quantile(0.20)]
        
        all_data["height"] = all_data["height"].apply(self.convert_height_to_inches)
        all_data = self.clean_team_abbreviations(all_data)
        
        required_cols = ["playerId", "name", "season", "position"]
        if 'icetime' in all_data.columns:
            required_cols.append('icetime')
        
        if 'team' in all_data.columns and 'team' not in required_cols:
            required_cols.append('team')
        
        numeric_cols = [col for col in all_data.columns 
                       if col not in required_cols 
                       and pd.api.types.is_numeric_dtype(all_data[col])
                       and col not in ['situation', 'birthDate']]
        
        all_data = all_data[required_cols + numeric_cols]
        
        self.all_numeric_features = numeric_cols
        return all_data
    
    def impute_data(self, all_data):
        preserve_cols = list(self.nonnum_columns)
        if 'icetime' in all_data.columns:
            preserve_cols = preserve_cols + ['icetime']
        
        if 'team' in all_data.columns and 'team' not in preserve_cols:
            preserve_cols.append('team')
        
        exclude_cols = set(preserve_cols)
        numeric_columns = [col for col in all_data.columns if col not in exclude_cols]
        
        if not numeric_columns:
            raise ValueError(f"No numeric columns found after filtering. Available columns: {list(all_data.columns)}")
        
        all_data_preserved = all_data[preserve_cols]
        all_data_numeric = all_data[numeric_columns]
        
        imputer = KNNImputer(n_neighbors=7)
        all_data_numeric = pd.DataFrame(
            imputer.fit_transform(all_data_numeric),
            columns=all_data_numeric.columns
        )
        
        if all_data_preserved.empty and all_data_numeric.empty:
            raise ValueError("No data to concatenate after imputation")
        
        if all_data_preserved.empty:
            all_data = all_data_numeric
        elif all_data_numeric.empty:
            all_data = all_data_preserved
        else:
            all_data = pd.concat([
                all_data_preserved.reset_index(drop=True),
                all_data_numeric.reset_index(drop=True)
            ], axis=1)
        all_data = self.add_bmi(all_data)
        return all_data