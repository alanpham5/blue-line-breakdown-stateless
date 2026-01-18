import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.data_loader import DataLoader
from utils.data_processor import DataProcessor
from utils.cache_manager import CacheManager
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

def upload_to_github_release(forwards_df, defensemen_df, token, repo):
    try:
        from github import Github
        import io
        
        g = Github(token)
        repo_obj = g.get_repo(repo)
        
        try:
            release = repo_obj.get_latest_release()
        except:
            release = repo_obj.create_git_release(
                tag=f"data-{pd.Timestamp.now().strftime('%Y%m%d')}",
                name=f"Data Update {pd.Timestamp.now().strftime('%Y-%m-%d')}",
                message="Daily automated data update",
                draft=False,
                prerelease=False
            )
        
        for df, filename in [(forwards_df, "forwards_processed.csv"), 
                             (defensemen_df, "defensemen_processed.csv")]:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            release.upload_asset(
                content_type="text/csv",
                name=filename,
                fileobj=io.BytesIO(csv_content.encode()),
                label=filename
            )
        
        return True
    except Exception as e:
        return False

def upload_to_s3(forwards_df, defensemen_df, bucket_name, aws_access_key, aws_secret_key):
    try:
        import boto3
        import io
        
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        for df, filename in [(forwards_df, "forwards_processed.csv"), 
                             (defensemen_df, "defensemen_processed.csv")]:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            s3.put_object(
                Bucket=bucket_name,
                Key=filename,
                Body=csv_buffer.getvalue().encode(),
                ContentType='text/csv'
            )
        
        return True
    except Exception as e:
        return False

def main():
    forwards_df, defensemen_df = process_data()
    
    cache_manager = CacheManager()
    cache_manager.save_processed_data(forwards_df, defensemen_df)
    
    upload_method = os.environ.get('UPLOAD_METHOD', 'github')
    
    if upload_method == 'github':
        token = os.environ.get('GITHUB_TOKEN')
        repo = os.environ.get('GITHUB_REPO')
        if token:
            upload_to_github_release(forwards_df, defensemen_df, token, repo)
    
    elif upload_method == 's3':
        bucket = os.environ.get('S3_BUCKET')
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        if bucket and access_key and secret_key:
            upload_to_s3(forwards_df, defensemen_df, bucket, access_key, secret_key)

if __name__ == '__main__':
    main()
