import pandas as pd
import os
from pathlib import Path

class CacheManager:
    def __init__(self, cache_dir=None):
        if cache_dir is None:
            script_dir = Path(__file__).parent.parent
            project_cache = script_dir / 'csv_cache'
            tmp_cache = Path('/tmp/csv_cache')
            
            if project_cache.exists() and (project_cache / 'forwards_processed.csv').exists():
                cache_dir = str(project_cache)
            elif tmp_cache.exists() and (tmp_cache / 'forwards_processed.csv').exists():
                cache_dir = str(tmp_cache)
            elif os.path.exists('/tmp') and os.access('/tmp', os.W_OK):
                cache_dir = '/tmp/csv_cache'
            else:
                cache_dir = 'csv_cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def get_cache_path(self, filename):
        return self.cache_dir / filename
    
    def _find_cache_file(self, filename):
        script_dir = Path(__file__).parent.parent
        project_cache = script_dir / 'csv_cache' / filename
        tmp_cache = Path('/tmp/csv_cache') / filename
        current_cache = self.cache_dir / filename
        
        if project_cache.exists():
            return project_cache
        elif tmp_cache.exists():
            return tmp_cache
        elif current_cache.exists():
            return current_cache
        return None
    
    def save_dataframe(self, df, filename):
        cache_path = self.get_cache_path(filename)
        df.to_csv(cache_path, index=False)
        return cache_path
    
    def load_dataframe(self, filename):
        cache_path = self._find_cache_file(filename)
        if cache_path and cache_path.exists():
            return pd.read_csv(cache_path)
        return None
    
    def cache_exists(self, filename):
        return self._find_cache_file(filename) is not None
    
    def save_processed_data(self, forwards_df, defensemen_df):
        self.save_dataframe(forwards_df, "forwards_processed.csv")
        self.save_dataframe(defensemen_df, "defensemen_processed.csv")

    def save_similarity_data(self, forwards_similarity_df, defensemen_similarity_df):
        self.save_dataframe(forwards_similarity_df, "forwards_similarity.csv")
        self.save_dataframe(defensemen_similarity_df, "defensemen_similarity.csv")

    def load_similarity_data(self):
        forwards = self.load_dataframe("forwards_similarity.csv")
        defensemen = self.load_dataframe("defensemen_similarity.csv")
        if forwards is not None and defensemen is not None:
            return forwards, defensemen
        return None, None
    
    def load_processed_data(self):
        forwards = self.load_dataframe("forwards_processed.csv")
        defensemen = self.load_dataframe("defensemen_processed.csv")
        if forwards is not None and defensemen is not None:
            return forwards, defensemen
        return None, None
