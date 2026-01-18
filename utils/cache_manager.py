import pandas as pd
import os
from pathlib import Path

class CacheManager:
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            if os.path.exists('/tmp') and os.access('/tmp', os.W_OK):
                cache_dir = '/tmp/csv_cache'
            else:
                cache_dir = 'csv_cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def get_cache_path(self, filename):
        return self.cache_dir / filename
    
    def save_dataframe(self, df, filename):
        cache_path = self.get_cache_path(filename)
        df.to_csv(cache_path, index=False)
        return cache_path
    
    def load_dataframe(self, filename):
        cache_path = self.get_cache_path(filename)
        if cache_path.exists():
            return pd.read_csv(cache_path)
        return None
    
    def cache_exists(self, filename):
        return self.get_cache_path(filename).exists()
    
    def save_processed_data(self, forwards_df, defensemen_df):
        self.save_dataframe(forwards_df, "forwards_processed.csv")
        self.save_dataframe(defensemen_df, "defensemen_processed.csv")
    
    def load_processed_data(self):
        forwards = self.load_dataframe("forwards_processed.csv")
        defensemen = self.load_dataframe("defensemen_processed.csv")
        if forwards is not None and defensemen is not None:
            return forwards, defensemen
        return None, None
