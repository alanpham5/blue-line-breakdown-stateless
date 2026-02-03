import pandas as pd
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.player_bio = None

    def _read_csv_response(self, response):
        try:
            return pd.read_csv(BytesIO(response.content), encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(BytesIO(response.content), encoding="utf-8-sig")
        
    def get_data(self, year):
        url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/skaters.csv"
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        return self._read_csv_response(resp)
    
    def load_player_bios(self):
        url = "https://moneypuck.com/moneypuck/playerData/playerBios/allPlayersLookup.csv"
        resp = requests.get(url, headers=self.headers, timeout=30)
        resp.raise_for_status()
        all_players = self._read_csv_response(resp)
        name_col = None
        for candidate in ['name', 'playerName', 'player_name']:
            if candidate in all_players.columns:
                name_col = candidate
                break
        bio_cols = ['playerId']
        if name_col:
            bio_cols.append(name_col)
        for col in ['height', 'weight', 'birthDate']:
            if col in all_players.columns:
                bio_cols.append(col)
        self.player_bio = all_players.loc[:, bio_cols].copy()
        if name_col and name_col != 'name':
            self.player_bio = self.player_bio.rename(columns={name_col: 'name'})
        return self.player_bio
    
    def load_all_seasons(self, start_year=2008, end_year=2026):
        if self.player_bio is None:
            self.load_player_bios()
        
        unmodded_data_d = {}
        unmodded_data_f = {}
        
        for year in range(start_year, end_year):
            try:
                data = self.get_data(year)
                data_d = data[data['position'] == 'D']
                data_f = data[data['position'] != 'D']
                unmodded_data_d[year] = data_d
                unmodded_data_f[year] = data_f
            except Exception as e:
                pass
        
        return unmodded_data_f, unmodded_data_d
