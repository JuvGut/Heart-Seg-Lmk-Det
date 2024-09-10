import boto3
import os
from botocore.exceptions import ClientError
from botocore.client import Config

s3 = boto3.client('s3',
    endpoint_url='https://dbe-lakefs.dbe.unibas.ch',
    aws_access_key_id=os.environ.get('LAKEFS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('LAKEFS_SECRET_ACCESS_KEY'),
    config=Config(verify=False)  # Temporarily disable SSL verification
)

bucket_name = 'heart-chambers-usb-ct'
branch_name = 'main'

def upload_to_lakefs(local_path, lakefs_path):
    full_lakefs_path = f"{branch_name}/{lakefs_path}"
    try:
        print(f"Uploading {local_path} to lakefs://{bucket_name}/{full_lakefs_path}")
        s3.upload_file(local_path, bucket_name, full_lakefs_path)
    except Exception as e:
        print(f"Error uploading {local_path}: {str(e)}")

# Configure the S3 client
# s3 = boto3.client('s3',
#     endpoint_url='https://dbe-lakefs.dbe.unibas.ch',
#     aws_access_key_id='AKIAJCV7SROWA72YI4SQ',
#     aws_secret_access_key='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'
# )

# aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
# aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

# s3 = boto3.client('s3',
#     endpoint_url='https://dbe-lakefs.dbe.unibas.ch',
#     aws_access_key_id=aws_access_key_id,
#     aws_secret_access_key=aws_secret_access_key
# )

# Set up directories
cache_dir = '/cache/juval.gutknecht'
project_dir = '/home/juval.gutknecht/Projects/heart-valve-segmentation'
os.makedirs(cache_dir, exist_ok=True)

# S3 bucket and branch information
bucket_name = 'heart-chambers-usb-ct'
branch_name = 'sync-datasets'  # New branch for syncing

def upload_to_lakefs(local_path, lakefs_path):
    """Upload file to lakeFS, skipping if it already exists"""
    try:
        # Check if file already exists in lakeFS
        s3.head_object(Bucket=bucket_name, Key=lakefs_path)
        print(f"File already exists in lakeFS: {lakefs_path}")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            # File doesn't exist, proceed with upload
            print(f"Uploading {local_path} to lakefs://{bucket_name}/{lakefs_path}")
            s3.upload_file(local_path, bucket_name, lakefs_path)
        else:
            # Some other error occurred
            print(f"Error checking {lakefs_path}: {e}")

def sync_directory(local_dir, lakefs_dir):
    """Sync a local directory with lakeFS"""
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            lakefs_path = os.path.join(lakefs_dir, relative_path).replace("\\", "/")
            upload_to_lakefs(local_path, lakefs_path)

def main():
    # Sync datasets
    datasets = [
        ("/home/juval.gutknecht/Projects/Data/A_Subset_012", "A_Subset_012"),
        ("/home/juval.gutknecht/Projects/Data/results", "results")
    ]

    for local_path, dataset_name in datasets:
        lakefs_dir = f"data/{dataset_name}"
        sync_directory(local_path, lakefs_dir)

    print("Sync completed successfully")

if __name__ == "__main__":
    main()