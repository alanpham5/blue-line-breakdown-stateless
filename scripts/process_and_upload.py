import sys
import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)

from utils.data_loader import DataLoader
from utils.data_processor import DataProcessor, prepare_similarity_data
import pandas as pd

def process_data():
    data_loader = DataLoader()
    data_processor = DataProcessor()
    
    forwards_raw, defensemen_raw = data_loader.load_all_seasons(2008, 2026)
    
    if not forwards_raw:
        raise ValueError("No forwards raw data loaded. Check data source and network connection.")
    if not defensemen_raw:
        raise ValueError("No defensemen raw data loaded. Check data source and network connection.")
    
    print(f"Loaded {len(forwards_raw)} years of forwards data and {len(defensemen_raw)} years of defensemen data", flush=True)
    
    forwards_processed = {}
    defensemen_processed = {}
    
    for year in forwards_raw.keys():
        try:
            f_data = data_processor.process_data(forwards_raw[year], data_loader.player_bio)
            d_data = data_processor.process_data(defensemen_raw[year], data_loader.player_bio)
            
            f_data = data_processor.impute_data(f_data)
            d_data = data_processor.impute_data(d_data)
            
            f_data = data_processor.scale_stats_per_60_min(f_data)
            d_data = data_processor.scale_stats_per_60_min(d_data)
            
            forwards_processed[year] = f_data
            defensemen_processed[year] = d_data
        except Exception as e:
            print(f"Error processing year {year}: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
    
    if not forwards_processed:
        raise ValueError("No forwards data processed. Check data loading and processing.")
    if not defensemen_processed:
        raise ValueError("No defensemen data processed. Check data loading and processing.")
    
    forwards_df = pd.concat(forwards_processed.values(), ignore_index=True)
    defensemen_df = pd.concat(defensemen_processed.values(), ignore_index=True)
    
    return forwards_df, defensemen_df

def upload_to_gcs(forwards_df, defensemen_df, forwards_similarity_df, defensemen_similarity_df, bucket_name, prefix="", project=None):
    try:
        from google.cloud import storage
        import io

        client = storage.Client(project=project) if project else storage.Client()
        bucket = client.bucket(bucket_name)

        clean_prefix = prefix.strip("/")

        def blob_name(filename):
            return f"{clean_prefix}/{filename}" if clean_prefix else filename

        upload_pairs = [
            (forwards_df, "forwards_processed.parquet"),
            (defensemen_df, "defensemen_processed.parquet"),
            (forwards_similarity_df, "forwards_similarity.parquet"),
            (defensemen_similarity_df, "defensemen_similarity.parquet"),
        ]
        for df, filename in upload_pairs:
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            blob = bucket.blob(blob_name(filename))
            blob.upload_from_string(parquet_buffer.getvalue(), content_type="application/octet-stream")

        return True
    except Exception as e:
        print(f"GCS upload error: {e}", flush=True)
        return False

def main():
    forwards_df, defensemen_df = process_data()

    forwards_similarity = prepare_similarity_data(forwards_df, normalization_method='standard', n_components=35)
    defensemen_similarity = prepare_similarity_data(defensemen_df, normalization_method='standard', n_components=35)
    
    bucket = os.environ.get('GCS_BUCKET')
    prefix = (os.environ.get('GCS_PREFIX', '') or '').strip()
    project = os.environ.get('GCP_PROJECT')
    if not bucket:
        raise ValueError("GCS_BUCKET is required.")
    ok = upload_to_gcs(forwards_df, defensemen_df, forwards_similarity, defensemen_similarity, bucket, prefix, project)
    if not ok:
        raise RuntimeError("GCS upload failed. Check credentials and bucket permissions.")
    print(f"Uploaded processed data to gs://{bucket}/{prefix}".rstrip("/"), flush=True)

if __name__ == '__main__':
    main()
