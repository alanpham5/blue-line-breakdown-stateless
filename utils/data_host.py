import os
from io import BytesIO

import pandas as pd
from google.cloud import storage

class DataHostManager:
    def __init__(self):
        self.gcs_bucket = os.environ.get('GCS_BUCKET')
        self.gcs_prefix = os.environ.get('GCS_PREFIX', '')
        self.gcp_project = os.environ.get('GCP_PROJECT')

    def _client(self):
        if not self.gcs_bucket:
            return None
        if self.gcp_project:
            return storage.Client(project=self.gcp_project)
        return storage.Client()

    def _blob_name(self, filename):
        prefix = (self.gcs_prefix or "").strip("/")
        if prefix:
            return f"{prefix}/{filename}"
        return filename

    def _read_parquet_bytes(self, content):
        return pd.read_parquet(BytesIO(content))

    def _download_blob_bytes(self, filename):
        client = self._client()
        if client is None:
            return None
        bucket = client.bucket(self.gcs_bucket)
        blob = bucket.blob(self._blob_name(filename))
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    def load_parquet(self, filename):
        try:
            content = self._download_blob_bytes(filename)
            if content is None:
                return None
            df = self._read_parquet_bytes(content)
            if df is None or df.empty:
                return None
            return df
        except Exception:
            return None
    
    def load_processed_data(self):
        forwards = self.load_parquet("forwards_processed.parquet")
        defensemen = self.load_parquet("defensemen_processed.parquet")
        
        if forwards is not None and defensemen is not None:
            return forwards, defensemen
        
        return None, None

    def load_similarity_data(self):
        forwards = self.load_parquet("forwards_similarity.parquet")
        defensemen = self.load_parquet("defensemen_similarity.parquet")
        if forwards is not None and defensemen is not None:
            return forwards, defensemen
        return None, None
    
    def check_data_available(self):
        try:
            client = self._client()
            if client is None:
                return False
            bucket = client.bucket(self.gcs_bucket)
            blob = bucket.blob(self._blob_name("forwards_processed.parquet"))
            return blob.exists()
        except Exception:
            return False
