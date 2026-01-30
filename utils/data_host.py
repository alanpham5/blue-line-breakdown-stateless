import pandas as pd
import requests
import os
from io import StringIO

class DataHostManager:
    def __init__(self):
        self.github_repo = os.environ.get('GITHUB_REPO')
        self.repo_base = f"https://github.com/{self.github_repo}/releases/download" if self.github_repo else ""
        self._latest_tag = None
        self._latest_release_data = None
    
    def get_latest_release_tag(self):
        if not self.github_repo:
            return None
        if self._latest_tag:
            return self._latest_tag
        
        try:
            api_url = f"https://api.github.com/repos/{self.github_repo}/releases/latest"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = requests.get(api_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                release_data = response.json()
                tag = release_data.get('tag_name')
                assets = release_data.get('assets', [])
                if tag:
                    self._latest_tag = tag
                    self._latest_release_data = release_data
                    return tag
        except Exception as e:
            pass
        return None
    
    def get_base_url(self):
        if self.github_repo and self.repo_base:
            return f"{self.repo_base}/latest"
        fallback = os.environ.get('DATA_HOST_URL', "")
        return fallback
    
    def get_release_by_tag(self, tag):
        if not self.github_repo:
            return None
        try:
            api_url = f"https://api.github.com/repos/{self.github_repo}/releases/tags/{tag}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token:
                headers['Authorization'] = f'token {github_token}'
            
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            pass
        return None
    
    def load_from_url(self, filename):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/csv,application/octet-stream,*/*'
        }
        
        if not self.github_repo or not self.repo_base:
            return None
        
        base_url = self.get_base_url()
        if not base_url:
            return None
        
        file_url = f"{base_url}/{filename}"
        
        try:
            response = requests.get(file_url, headers=headers, timeout=30, allow_redirects=True)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            if df.empty:
                return None
            return df
        except requests.exceptions.HTTPError as e:
            return None
        except requests.exceptions.RequestException:
            return None
        except Exception:
            return None
    
    def load_processed_data(self):
        forwards = self.load_from_url("forwards_processed.csv")
        defensemen = self.load_from_url("defensemen_processed.csv")
        
        if forwards is not None and defensemen is not None:
            return forwards, defensemen
        
        return None, None

    def load_similarity_data(self):
        forwards = self.load_from_url("forwards_similarity.csv")
        defensemen = self.load_from_url("defensemen_similarity.csv")
        if forwards is not None and defensemen is not None:
            return forwards, defensemen
        return None, None
    
    def check_data_available(self):
        try:
            base_url = self.get_base_url()
            response = requests.head(f"{base_url}/forwards_processed.csv", timeout=5)
            return response.status_code == 200
        except:
            return False
