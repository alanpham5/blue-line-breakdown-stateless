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
            "OnIce_F_xGoals", "OnIce_F_goals", "OnIce_F_shotAttempts",
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
        GOALS_PER_WIN = 4.5



        EV_OFF_WEIGHT = 2.5
        EV_DEF_WEIGHT = 2
        PP_OFF_WEIGHT = 1
        PK_DEF_WEIGHT = 0.5

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
                if key in ['timeOnIce', 'icetime']:
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

        df['EV_min'] = df[ev_min_col] if ev_min_col else 0
        df['PP_min'] = df[pp_min_col] if pp_min_col else 0
        df['PK_min'] = df[pk_min_col] if pk_min_col else 0

        df['Total_min'] = (df['EV_min'] + df['PP_min'] + df['PK_min']).clip(lower=1)
        df['PP_weight'] = df['PP_min'] / df['Total_min']
        df['PK_weight'] = df['PK_min'] / df['Total_min']

        xgf_ev_cols = ['OnIce_F_xGoals', 'xGoalsForOnIceAdjusted', 'xGoalsForOnIce']
        xga_ev_cols = ['OnIce_A_xGoals', 'xGoalsAgainstOnIceAdjusted', 'xGoalsAgainstOnIce']

        xgf_ev_col = next((c for c in xgf_ev_cols if c in df.columns), None)
        xga_ev_col = next((c for c in xga_ev_cols if c in df.columns), None)

        if xgf_ev_col and xga_ev_col:
            df['xGF_EV_60'] = (df[xgf_ev_col] / df['EV_min']) * 60 
            df['xGA_EV_60'] = (df[xga_ev_col] / df['EV_min']) * 60
        else:
            df['xGF_EV_60'] = 0.0
            df['xGA_EV_60'] = 0.0

        REP_XGF_EV = df['xGF_EV_60'].quantile(0.3)
        REP_XGA_EV = df['xGA_EV_60'].quantile(0.7)
        REP_XGF_PP = ((df['OnIce_F_xGoals'] / df['PP_min']) * 60).quantile(0.5)
        REP_XGA_PK = ((df['OnIce_A_xGoals'] / df['PK_min']) * 60).quantile(0.5)

        df['Off_GAR'] = (
            (df['xGF_EV_60'] - REP_XGF_EV)
            * EV_OFF_WEIGHT
        )

        df['Def_GAR'] = (
            (REP_XGA_EV - df['xGA_EV_60'])
            * EV_DEF_WEIGHT
        )
        df['Def_GAR'] = df['Def_GAR']

        df['PP_GAR'] = (
            ((df['OnIce_F_xGoals'] / df['PP_min']) * 60 - REP_XGF_PP)
            * df['PP_weight']
            * PP_OFF_WEIGHT
        )



        df['PK_GAR'] = (
            (REP_XGA_PK - (df['OnIce_A_xGoals'] / df['PK_min']) * 60)
            * df['PK_weight']
            * PK_DEF_WEIGHT
        )

        # "Shelter tax": tax d-men that mainly play on PP and sheltered minutes

        pp_share = df['PP_min'] / df['icetime']
        pk_share = df['PK_min'] / df['icetime']


        pp_share_quantile = pp_share[df['position'] == 'D'].quantile(0.80)
        pk_share_quantile = pk_share[df['position'] == 'D'].quantile(0.3)


        sheltered_mask = (df['position'] == 'D') & (pp_share >= pp_share_quantile) & (pk_share <= pk_share_quantile)

        pp1_minutes_ref = df.loc[df['position'] == 'D', 'PP_min'].quantile(0.85)
        rel_pp_min = df['PP_min'] / (pp1_minutes_ref + 1e-6)


        share_gate = sheltered_mask.astype(float)
        shelter_index = share_gate * (rel_pp_min) ** 7


        for col in ['Off_GAR', 'PP_GAR', 'Def_GAR']:
            pos_mask = df[col] > 0
            neg_mask = ~pos_mask
            df.loc[sheltered_mask & pos_mask, col] *= (1 - (0.1 * shelter_index[sheltered_mask]**1.2))
            df.loc[sheltered_mask & neg_mask, col] *= (1 + (0.1 * shelter_index[sheltered_mask]**1.2))


        penalty_drawn_cols = ['penaltyMinutesDrawn', 'penaltiesDrawn', 'penaltiesDrawnEV']
        penalty_taken_cols = ['penalityMinutes', 'penaltiesTaken', 'penaltiesTakenEV']

        penalty_drawn_col = next((c for c in penalty_drawn_cols if c in df.columns), None)
        penalty_taken_col = next((c for c in penalty_taken_cols if c in df.columns), None)

        df['Penalty_GAR'] = (
            (df[penalty_drawn_col] - df[penalty_taken_col]) * 0.05
        ).fillna(0) if penalty_drawn_col and penalty_taken_col else 0

        faceoff_won_cols = ['faceOffsWon', 'faceoffsWon']
        faceoff_lost_cols = ['faceOffsLost', 'faceoffsLost']

        faceoff_won_col = next((c for c in faceoff_won_cols if c in df.columns), None)
        faceoff_lost_col = next((c for c in faceoff_lost_cols if c in df.columns), None)

        df['Faceoff_GAR'] = (
            (df[faceoff_won_col] - (df[faceoff_won_col] + df[faceoff_lost_col]) / 2) * 0.005
        ).fillna(0) if faceoff_won_col and faceoff_lost_col else 0

        df['Total_GAR'] = (
            df['Off_GAR'] +
            df['Def_GAR'] +
            df['PP_GAR'] +
            df['PK_GAR'] +
            df['Penalty_GAR'] +
            df['Faceoff_GAR']
        )

        df['WAR_scaled'] = df['Total_GAR'] / GOALS_PER_WIN

        if 'gameScore' in df.columns:
            df['gameScore_clean'] = df['gameScore'].fillna(0).clip(-100, 100)
        else:
            df['gameScore_clean'] = 0.0

        df['WAR_scaled'] = df['WAR_scaled'] * df['icetime']
        df['gameScore_clean'] = df['gameScore_clean'] * df['icetime']

        replacement_level_war = df['WAR_scaled'].median()
        replacement_level_gs = df['gameScore_clean'].median()

        df['WAR'] = (
            df['WAR_scaled']
            - replacement_level_war
            + 0.15 * (df['gameScore_clean'] - replacement_level_gs)
        )


        tau = df['icetime'].median()   # trust threshold
        confidence = (df['icetime'] / (df['icetime'] + tau))
        df['WAR'] = df['WAR'] * confidence
 


        df = df.drop(
            columns=[
                'WAR_scaled',
                'gameScore_clean',
                'Total_min',
                'PP_weight',
                'PK_weight',
                'xGF_EV_60',
                'xGA_EV_60',
                'EV_share'
            ],
            errors='ignore'
        )

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

        df = df.copy()
        df['icetime_hours'] = df['icetime'] / 3600


        never_scale_cols = set([
            "playerId", "name", "season", "team",
            "height", "weight", "age", "bmi", "position",
            "icetime", "icetime_hours",
            "onIce_corsiPercentage",
            "timeOnIcePP", "timeOnIcePK", "timeOnIceEV"
        ])
        never_scale_cols.update(self.war_columns)

        possible_total_cols = [
            'games_played',
            'I_F_goals', 'I_F_primaryAssists', 'I_F_secondaryAssists', 'I_F_points',
            'penalityMinutes', 'pim', 'penalties', 'penaltiesDrawn', 'penaltiesTaken'
        ]
        total_stats_cols = []
        for col in possible_total_cols:
            if col in df.columns and f'{col}_total' not in df.columns:
                df[f'{col}_total'] = df[col].copy()
                total_stats_cols.append(f'{col}_total')

        never_scale_cols.update(total_stats_cols)

        if not any(c in df.columns for c in ['games_played_total', 'games_total', 'GP_total']):
            if all(c in df.columns for c in ['timeOnIcePP', 'timeOnIcePK', 'timeOnIceEV']):
                total_icetime_minutes = (df['timeOnIcePP'] + df['timeOnIcePK'] + df['timeOnIceEV']) / 60.0
                df['games_played_total'] = (total_icetime_minutes / 15.0).round().astype(int).clip(lower=1)
                never_scale_cols.add('games_played_total')


        stats_columns = [col for col in df.select_dtypes(include="number").columns
                        if col not in never_scale_cols]

        if stats_columns:
            df[stats_columns] = df[stats_columns].div(df['icetime_hours'], axis=0)


        df = df.drop(columns=['icetime_hours', 'icetime'], errors='ignore')

        return df
    
    def process_data(self, df, player_bio):
        merged_data = self.merge_player_data(df, player_bio)
        all_data = merged_data[merged_data['situation'] == 'all'].copy()

        pp_data = merged_data[merged_data['situation'] == '5on4'][['playerId', 'season', 'icetime']].rename(columns={'icetime': 'timeOnIcePP'})
        all_data = all_data.merge(pp_data, on=['playerId', 'season'], how='left')

        pk_data = merged_data[merged_data['situation'] == '4on5'][['playerId', 'season', 'icetime']].rename(columns={'icetime': 'timeOnIcePK'})
        all_data = all_data.merge(pk_data, on=['playerId', 'season'], how='left')

        ev_data = merged_data[merged_data['situation'] == '5on5'][['playerId', 'season', 'icetime']].rename(columns={'icetime': 'timeOnIceEV'})
        all_data = all_data.merge(ev_data, on=['playerId', 'season'], how='left')


        all_data['timeOnIcePP'] = all_data['timeOnIcePP'].fillna(0)
        all_data['timeOnIcePK'] = all_data['timeOnIcePK'].fillna(0)
        all_data['timeOnIceEV'] = all_data['timeOnIceEV'].fillna(0)


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