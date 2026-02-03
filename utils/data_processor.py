import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


SIMILARITY_EXCLUDE_COLS = {
    "playerId", "name", "season", "team",
    "EV_min", "PP_min", "PK_min",
    "timeOnIceEV", "timeOnIcePP", "timeOnIcePK",
    "Total_GAR"
}


def prepare_similarity_data(df, normalization_method='standard', n_components=35, feature_weights=None):

    from utils.similarity_engine import calculate_feature_weights as get_feature_weights

    df_out = df.copy()
    nonnum_columns = ["playerId", "name", "position", "season", "team"]

    if "team" in df_out.columns:
        pass
    else:
        nonnum_columns = [c for c in nonnum_columns if c in df_out.columns]

    numeric_columns = [
        col for col in df_out.columns
        if col not in SIMILARITY_EXCLUDE_COLS and col not in nonnum_columns
        and pd.api.types.is_numeric_dtype(df_out[col])
    ]
    if not numeric_columns:
        raise ValueError("No numeric columns available for similarity preparation")


    seasons = df_out["season"].unique()
    normalized_dfs = []
    for season in seasons:
        season_df = df_out[df_out["season"] == season].copy()
        scaler = StandardScaler()
        season_df[numeric_columns] = scaler.fit_transform(season_df[numeric_columns])
        normalized_dfs.append(season_df)
    df_out = pd.concat(normalized_dfs, axis=0).sort_index()


    df_nonnum = df_out[[c for c in nonnum_columns if c in df_out.columns]].copy()
    df_numeric = df_out[numeric_columns].copy()
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=-1e6, upper=1e6)
    scaler = StandardScaler()
    df_numeric_scaled = scaler.fit_transform(df_numeric)

    if feature_weights is None:
        feature_weights = get_feature_weights(numeric_columns)
    weights_array = np.array([feature_weights.get(col, 1.0) for col in numeric_columns])
    df_numeric_scaled = df_numeric_scaled * weights_array

    df_numeric_values = np.nan_to_num(
        df_numeric_scaled.astype(np.float64),
        nan=0.0, posinf=100.0, neginf=-100.0
    )
    df_numeric_values = np.clip(df_numeric_values, -100, 100)

    n_samples, n_features = df_numeric_values.shape
    n_comp = min(n_components, n_features, n_samples)
    if n_comp < 1:
        raise ValueError("Not enough components for PCA")

    try:
        pca = PCA(n_components=n_comp)
        trans_df = pca.fit_transform(df_numeric_values)
    except (ValueError, np.linalg.LinAlgError):
        df_numeric_values = np.clip(df_numeric_values, -50, 50)
        pca = PCA(n_components=n_comp, svd_solver='arpack')
        trans_df = pca.fit_transform(df_numeric_values)

    pc_cols = list(range(n_comp))
    trans_df = pd.DataFrame(trans_df, index=df_out.index, columns=pc_cols)
    result = pd.concat([df_nonnum.reset_index(drop=True), trans_df.reset_index(drop=True)], axis=1)
    return result


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
        
        df_stats_copy = df_stats.copy()
        if 'name' in df_stats_copy.columns and 'name' in df_info.columns:
            df_stats_copy = df_stats_copy.drop(columns=['name'])
        
        merged_df = pd.merge(df_stats_copy, df_info, on='playerId', how='inner')
        merged_df['age'] = merged_df.apply(self.calculate_age, axis=1)
        
        accent_name_overrides = {
            8482116: 'Tim Stützle',
            8481535: 'Nils Höglander',
            8475825: 'Jani Hakanpää',
            8482109: 'Alexis Lafrenière',
            8480796: 'Martin Fehérváry',
            8476882: 'Teuvo Teräväinen',
            8475714: 'Calle Järnkrok',
            8477416: 'Oliver Björkstrand',
            8471262: 'Carl Söderberg',
            8475175: 'Magnus Pääjärvi',
            8481813: 'Gaëtan Haas',
            8477944: 'Jakub Vrána',
            8477956: 'David Pastrňák',
            8476292: 'Ondřej Palát',
            8480039: 'Martin Nečas',
            8474161: 'Jakub Voráček',
            8476881: 'Tomáš Hertl',
            8475193: 'Tomáš Tatar',
            8469521: 'Tomáš Plekanec',
            8475765: 'Vladimír Tarasenko',
            8465009: 'Zdeno Chára',
            8466148: 'Marián Hossa',
            8459534: 'Miroslav Šatan',
            8483495: 'Šimon Nemec',
            8478416: 'Erik Černák',
            8478870: 'Rūdolfs Balcers',
            8479022: 'Rodrigo Ābols',
            8479729: 'Kristiāns Rubīns',
            8478007: 'Elvis Merzļikins',
            8481668: 'Artūrs Šilovs',
            8473484: 'Kaspars Daugaviņš',
            8477930: 'Pierre-Édouard Bellemare',
            8467400: 'François Beauchemin',
            8477520: 'Jean-Sébastien Dea',
            8477444: 'André Burakovsky',
        }
        for player_id, name in accent_name_overrides.items():
            merged_df.loc[merged_df['playerId'] == player_id, 'name'] = name

        columns = ['playerId', 'name', 'height', 'weight', 'age'] + \
                  [col for col in df_stats_copy.columns if col not in ['playerId']]
        return merged_df[columns]
    
    def add_bmi(self, df):
        df['bmi'] = df['weight'] / (df['height'] ** 2) * 703
        return df
    
    def calculate_war(self, df):
        GOALS_PER_WIN = 4.5

        EV_OFF_WEIGHT = 1.83
        EV_DEF_WEIGHT = 1.83
        PP_OFF_WEIGHT = 0.67
        PK_DEF_WEIGHT = 0.67

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

        df['EV_min'] = df['EV_min'].fillna(0)
        df['PP_min'] = df['PP_min'].fillna(0)
        df['PK_min'] = df['PK_min'].fillna(0)

        df['Total_min'] = df['EV_min'] + df['PP_min'] + df['PK_min']
        df['PP_weight'] = df['PP_min'] / df['Total_min']
        df['PK_weight'] = df['PK_min'] / df['Total_min']

        xgf_ev_cols = ['OnIce_F_xGoals', 'xGoalsForOnIceAdjusted', 'xGoalsForOnIce']
        xga_ev_cols = ['OnIce_A_xGoals', 'xGoalsAgainstOnIceAdjusted', 'xGoalsAgainstOnIce']
        xgf_pp_cols = ['OnIce_F_xGoals_PP', 'OnIce_F_xGoals', 'xGoalsForOnIceAdjusted', 'xGoalsForOnIce']
        xga_pk_cols = ['OnIce_A_xGoals_PK','OnIce_A_xGoals', 'xGoalsAgainstOnIceAdjusted', 'xGoalsAgainstOnIce']

        xgf_ev_col = next((c for c in xgf_ev_cols if c in df.columns), None)
        xga_ev_col = next((c for c in xga_ev_cols if c in df.columns), None)
        xgf_pp_col = next((c for c in xgf_pp_cols if c in df.columns), None)
        xga_pk_col = next((c for c in xga_pk_cols if c in df.columns), None)

        if xgf_ev_col and xga_ev_col:
            df['xGF_EV_60'] = ((df[xgf_ev_col] / df['EV_min']) * 3600).fillna(0)
            df['xGA_EV_60'] = ((df[xga_ev_col] / df['EV_min']) * 3600).fillna(0)
            df['xGF_PP_60'] = ((df[xgf_pp_col] / df['PP_min']) * 60).fillna(0)
            df['xGA_PK_60'] = ((df[xga_pk_col] / df['PK_min']) * 60).fillna(0)
        else:
            df['xGF_EV_60'] = 0.0
            df['xGA_EV_60'] = 0.0
            df['xGF_PP_60'] = 0
            df['xGA_PK_60'] = 0

        REP_XGF_EV = df['xGF_EV_60'].quantile(0.3)
        REP_XGA_EV = df['xGA_EV_60'].quantile(0.7)
        REP_XGF_PP = df['xGF_PP_60'].quantile(0.5)
        REP_XGA_PK = df['xGA_PK_60'].quantile(0.5)

        df['Off_GAR'] = (
            (df['xGF_EV_60'] - REP_XGF_EV)
            * EV_OFF_WEIGHT
        ).fillna(0)

        df['Def_GAR'] = (
            (REP_XGA_EV - df['xGA_EV_60'])
            * EV_DEF_WEIGHT
        ).fillna(0)

        df['PP_GAR'] = (
            (df['xGF_PP_60'] - REP_XGF_PP)
            * df['PP_weight']
            * PP_OFF_WEIGHT
        ).fillna(0)

        df['PK_GAR'] = (
            (REP_XGA_PK - df['xGA_PK_60'])
            * df['PK_weight']
            * PK_DEF_WEIGHT
        ).fillna(0)

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

        replacement_level_war = df['WAR_scaled'].quantile(0.45)
        replacement_level_gs = df['gameScore_clean'].quantile(0.45)

        df['WAR'] = (
            df['WAR_scaled']
            - replacement_level_war
            + 0.15 * (df['gameScore_clean'] - replacement_level_gs)
        )

        # Apply shelter tax to defensemen
        pp_share = df['PP_min'] / df['icetime']
        pk_share = df['PK_min'] / df['icetime']

        pp_share_quantile = pp_share[df['position'] == 'D'].quantile(0.80)
        pk_share_quantile = pk_share[df['position'] == 'D'].quantile(0.3)

        sheltered_mask = (df['position'] == 'D') & (pp_share >= pp_share_quantile) & (pk_share <= pk_share_quantile)

        pp_min_sheltered = df.loc[sheltered_mask, 'PP_min']
        if len(pp_min_sheltered) > 0:
            pp_min_p25 = pp_min_sheltered.quantile(0.25)
            pp_min_max = pp_min_sheltered.max()
            
            rel_pp_normalized = (df['PP_min'] - pp_min_p25) / (pp_min_max - pp_min_p25)
            rel_pp_normalized = np.clip(rel_pp_normalized, 0, 1)
            
            def calculate_shelter_index(x):
                return np.where(
                    x <= 0.75,
                    x + 0.15,  
                    0.76 + 0.1 * (x - 0.7)  
                )
            
            shelter_index = sheltered_mask.astype(float) * calculate_shelter_index(rel_pp_normalized)
            
            positive_war_mask = sheltered_mask & (df['WAR'] > 0)
            df.loc[positive_war_mask, 'WAR'] *= (1 - shelter_index[positive_war_mask])


        # Scale WAR to [-8, 8] while preserving 0
        war_positive_max = df.loc[df['WAR'] > 0, 'WAR'].max() if (df['WAR'] > 0).any() else 1
        war_negative_min = df.loc[df['WAR'] < 0, 'WAR'].min() if (df['WAR'] < 0).any() else -1

        if war_positive_max > 8:
            positive_mask = df['WAR'] > 0
            df.loc[positive_mask, 'WAR'] = df.loc[positive_mask, 'WAR'] * (20 / war_positive_max)


        if war_negative_min < -8:
            negative_mask = df['WAR'] < 0
            df.loc[negative_mask, 'WAR'] = df.loc[negative_mask, 'WAR'] * (4 / abs(war_negative_min))

        # Cleanup temporary columns
        df = df.drop(
            columns=[
                'WAR_scaled',
                'gameScore_clean',
                'Total_min',
                'PP_weight',
                'PK_weight',
                'xGF_EV_60',
                'xGA_EV_60',
                'xGF_PP_60',
                'xGA_PK_60',
                'EV_min',
                'PP_min',
                'PK_min',
                'EV_share',
                'OnIce_F_xGoals_PP',
                'OnIce_A_xGoals_PK'
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

        pp_data = merged_data[merged_data['situation'] == '5on4'][['playerId', 'season', 'icetime', 'OnIce_F_xGoals']].rename(columns={'icetime': 'timeOnIcePP', 'OnIce_F_xGoals': 'OnIce_F_xGoals_PP'})
        all_data = all_data.merge(pp_data, on=['playerId', 'season'], how='left')

        pk_data = merged_data[merged_data['situation'] == '4on5'][['playerId', 'season', 'icetime', 'OnIce_A_xGoals']].rename(columns={'icetime': 'timeOnIcePK', 'OnIce_A_xGoals': 'OnIce_A_xGoals_PK'})
        all_data = all_data.merge(pk_data, on=['playerId', 'season'], how='left')

        ev_data = merged_data[merged_data['situation'] == '5on5'][['playerId', 'season', 'icetime']].rename(columns={'icetime': 'timeOnIceEV'})
        all_data = all_data.merge(ev_data, on=['playerId', 'season'], how='left')


        all_data['timeOnIcePP'] = all_data['timeOnIcePP'].fillna(0)
        all_data['OnIce_F_xGoals_PP'] = all_data['OnIce_F_xGoals_PP'].fillna(0)
        all_data['timeOnIcePK'] = all_data['timeOnIcePK'].fillna(0)
        all_data['OnIce_A_xGoals_PK'] = all_data['OnIce_A_xGoals_PK'].fillna(0)
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
