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
            "I_F_takeaways", "I_F_giveaways", "I_F_shotsBlockedByPlayer", "shotsBlockedByPlayer",
            "OnIce_F_xGoals", "OnIce_F_goals",
            "OnIce_A_xGoals", "OnIce_A_goals"
        ]
        self.nonnum_columns = ["playerId", "name", "position", "season"]
    
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
    
    def scale_stats_per_60_min(self, df):
        if 'icetime' not in df.columns:
            return df
        
        df['icetime_hours'] = df['icetime'] / 3600
        stats_columns = [col for col in df.columns if col not in 
                        ["playerId", "name", "season", 'height', 'weight', 
                         "age", "bmi", "position", "icetime"]]
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
        
        available_features = [f for f in self.icetime_relative_features if f in all_data.columns]
        
        if 'icetime' in all_data.columns:
            available_features.append('icetime')
        
        all_data = all_data.loc[:, available_features]
        all_data["height"] = all_data["height"].apply(self.convert_height_to_inches)
        return all_data
    
    def impute_data(self, all_data):
        preserve_cols = list(self.nonnum_columns)
        if 'icetime' in all_data.columns:
            preserve_cols = preserve_cols + ['icetime']
        exclude_cols = set(preserve_cols)
        numeric_columns = [col for col in all_data.columns if col not in exclude_cols]
        all_data_preserved = all_data[preserve_cols]
        all_data_numeric = all_data[numeric_columns]
        
        imputer = KNNImputer(n_neighbors=7)
        all_data_numeric = pd.DataFrame(
            imputer.fit_transform(all_data_numeric),
            columns=all_data_numeric.columns
        )
        
        all_data = pd.concat([
            all_data_preserved.reset_index(drop=True),
            all_data_numeric.reset_index(drop=True)
        ], axis=1)
        all_data = self.add_bmi(all_data)
        return all_data