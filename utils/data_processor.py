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
        self.war_columns = [
            'EV_min', 'PP_min', 'PK_min', 'Off_GAR', 'Def_GAR', 
            'PP_GAR', 'PK_GAR', 'Penalty_GAR', 'Faceoff_GAR', 
            'Total_GAR', 'WAR'
        ]
        
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
    
    def calculate_war(self, df):
        # I did not create this formula. I used the internet and some AI to form it.
        GOALS_PER_WIN = 4.5
        
        REP_XGF_EV = 1.079
        REP_XGA_EV = 1.484
        REP_XGF_PP = 0.949
        REP_XGA_PK = 3.44
        
        df = df.copy()
        
        time_cols_map = {
            'timeOnIce': 'timeOnIce',
            'icetime': 'icetime',
            'timeOnIceEV': 'timeOnIceEV',
            'timeOnIcePP': 'timeOnIcePP',
            'timeOnIcePK': 'timeOnIcePK'
        }
        
        ev_min_col = None
        pp_min_col = None
        pk_min_col = None
        
        for key, col in time_cols_map.items():
            if col in df.columns:
                if key == 'timeOnIce' or key == 'icetime':
                    if ev_min_col is None:
                        ev_min_col = col
                elif key == 'timeOnIceEV':
                    ev_min_col = col
                elif key == 'timeOnIcePP':
                    pp_min_col = col
                elif key == 'timeOnIcePK':
                    pk_min_col = col
        
        if ev_min_col is None:
            for c in self.war_columns:
                if c not in df.columns:
                    df[c] = 0.0
            return df
        
        if ev_min_col == 'icetime':
            df['EV_min'] = df[ev_min_col] / 60
        else:
            df['EV_min'] = df[ev_min_col]
        
        if pp_min_col:
            if pp_min_col == 'icetime':
                df['PP_min'] = df[pp_min_col] / 60
            else:
                df['PP_min'] = df[pp_min_col]
        else:
            df['PP_min'] = 0
        
        if pk_min_col:
            if pk_min_col == 'icetime':
                df['PK_min'] = df[pk_min_col] / 60
            else:
                df['PK_min'] = df[pk_min_col]
        else:
            df['PK_min'] = 0
        
        xgf_ev_cols = ['xGoalsForOnIceAdjusted', 'OnIce_F_xGoals', 'xGoalsForOnIce']
        xga_ev_cols = ['xGoalsAgainstOnIceAdjusted', 'OnIce_A_xGoals', 'xGoalsAgainstOnIce']
        xgf_pp_cols = ['xGoalsForOnIcePP', 'xGoalsForPP']
        xga_pk_cols = ['xGoalsAgainstOnIcePK', 'xGoalsAgainstPK']
        
        xgf_ev_col = next((col for col in xgf_ev_cols if col in df.columns), None)
        xga_ev_col = next((col for col in xga_ev_cols if col in df.columns), None)
        xgf_pp_col = next((col for col in xgf_pp_cols if col in df.columns), None)
        xga_pk_col = next((col for col in xga_pk_cols if col in df.columns), None)
        
        if xgf_ev_col:
            df['Off_GAR'] = ((df[xgf_ev_col] - REP_XGF_EV) * df['EV_min'] / 60).fillna(0)
        else:
            df['Off_GAR'] = 0
        
        if xga_ev_col:
            df['Def_GAR'] = ((REP_XGA_EV - df[xga_ev_col]) * df['EV_min'] / 60).fillna(0)
        else:
            df['Def_GAR'] = 0
        
        if xgf_pp_col and pp_min_col:
            df['PP_GAR'] = ((df[xgf_pp_col] - REP_XGF_PP) * df['PP_min'] / 60).fillna(0)
        else:
            df['PP_GAR'] = 0
        
        if xga_pk_col and pk_min_col:
            df['PK_GAR'] = ((REP_XGA_PK - df[xga_pk_col]) * df['PK_min'] / 60).fillna(0)
        else:
            df['PK_GAR'] = 0
        
        penalty_drawn_cols = ['penaltiesDrawn', 'penaltiesDrawnEV']
        penalty_taken_cols = ['penaltiesTaken', 'penaltiesTakenEV']
        
        penalty_drawn_col = next((col for col in penalty_drawn_cols if col in df.columns), None)
        penalty_taken_col = next((col for col in penalty_taken_cols if col in df.columns), None)
        
        if penalty_drawn_col and penalty_taken_col:
            df['Penalty_GAR'] = ((df[penalty_drawn_col] - df[penalty_taken_col]) * 0.2).fillna(0)
        else:
            df['Penalty_GAR'] = 0
        
        faceoff_won_cols = ['faceOffsWon', 'faceoffsWon']
        faceoff_expected_cols = ['expectedFaceOffsWon', 'expectedFaceoffsWon']
        
        faceoff_won_col = next((col for col in faceoff_won_cols if col in df.columns), None)
        faceoff_expected_col = next((col for col in faceoff_expected_cols if col in df.columns), None)
        
        if faceoff_won_col and faceoff_expected_col:
            df['Faceoff_GAR'] = ((df[faceoff_won_col] - df[faceoff_expected_col]) * 0.01).fillna(0)
        else:
            df['Faceoff_GAR'] = 0
        
        df['Total_GAR'] = (
            df['Off_GAR'] + df['Def_GAR'] + df['PP_GAR'] + 
            df['PK_GAR'] + df['Penalty_GAR'] + df['Faceoff_GAR']
        )
        
        df['WAR'] = df['Total_GAR'] / GOALS_PER_WIN
        
        return df
    
    def clean_team_abbreviations(self, df):
        if 'team' not in df.columns:
            return df
        
        df = df.copy()
        df['team'] = df['team'].astype(str).replace(self.team_abbrev_cleanup).replace('nan', np.nan)
        return df
    
    def scale_stats_per_60_min(self, df):
        if 'icetime' not in df.columns:
            return df
        
        df['icetime_hours'] = df['icetime'] / 3600
        
        exclude_cols = ["playerId", "name", "season", 'height', 'weight', 
                       "age", "bmi", "position", "icetime", "onIce_corsiPercentage",
                       "icetime_hours", "timeOnIcePP", "timeOnIcePK", "timeOnIceEV"]
        
        exclude_cols.append('team')
        exclude_cols.extend(self.war_columns)
        
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
        all_data = merged_data[merged_data['situation'] == 'all'].copy()

        # Calculate timeOnIce for different situations
        pp_data = merged_data[merged_data['situation'] == '5on4'][['playerId', 'season', 'icetime']].rename(columns={'icetime': 'timeOnIcePP'})
        all_data = all_data.merge(pp_data, on=['playerId', 'season'], how='left')

        pk_data = merged_data[merged_data['situation'] == '4on5'][['playerId', 'season', 'icetime']].rename(columns={'icetime': 'timeOnIcePK'})
        all_data = all_data.merge(pk_data, on=['playerId', 'season'], how='left')

        ev_data = merged_data[merged_data['situation'] == '5on5'][['playerId', 'season', 'icetime']].rename(columns={'icetime': 'timeOnIceEV'})
        all_data = all_data.merge(ev_data, on=['playerId', 'season'], how='left')

        # Fill missing times with 0
        all_data['timeOnIcePP'] = all_data['timeOnIcePP'].fillna(0)
        all_data['timeOnIcePK'] = all_data['timeOnIcePK'].fillna(0)
        all_data['timeOnIceEV'] = all_data['timeOnIceEV'].fillna(0)

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
        all_data = self.calculate_war(all_data)
        return all_data