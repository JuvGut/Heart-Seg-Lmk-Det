import boto3
import os
from botocore.exceptions import ClientError
from botocore.client import Config

# Konfigurationsvariablen
data_repository = "heart-chambers-usb-ct"
branch = "dev"
cache_path = "/cache/juval.gutknecht/heart-chambers-usb-ct"
s3_endpoint = "https://dbe-lakefs.dbe.unibas.ch:8000"
access_key = "AKIAJCV7SROWA72YI4SQ"
secret_key = "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"

# S3 Client konfigurieren
s3 = boto3.client('s3',
    endpoint_url=s3_endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version='s3v4')
)

def upload_to_lakefs(local_path, lakefs_path):
    """Upload file to lakeFS, skipping if it already exists"""
    full_lakefs_path = f"{branch}/dataset/{lakefs_path}"
    try:
        # Überprüfen, ob die Datei bereits in lakeFS existiert
        s3.head_object(Bucket=data_repository, Key=full_lakefs_path)
        print(f"Datei existiert bereits in lakeFS: {full_lakefs_path}")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            # Datei existiert nicht, Upload durchführen
            print(f"Uploading {local_path} to lakefs://{data_repository}/{full_lakefs_path}")
            s3.upload_file(local_path, data_repository, full_lakefs_path)
        else:
            # Ein anderer Fehler ist aufgetreten
            print(f"Fehler beim Überprüfen von {full_lakefs_path}: {e}")

def sync_directory(local_dir, lakefs_dir):
    """Synchronisiere ein lokales Verzeichnis mit lakeFS"""
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            lakefs_path = os.path.join(lakefs_dir, relative_path).replace("\\", "/")
            upload_to_lakefs(local_path, lakefs_path)

def main():
    # Verzeichnisse erstellen
    os.makedirs(cache_path, exist_ok=True)

    # Datasets synchronisieren
    datasets = [
        ("/home/juval.gutknecht/Projects/Data/A_Subset_012", "A_Subset_012"),
        ("/home/juval.gutknecht/Projects/Data/results", "results")
    ]

    for local_path, dataset_name in datasets:
        lakefs_dir = f"data/{dataset_name}"
        sync_directory(local_path, lakefs_dir)

    print("Synchronisation erfolgreich abgeschlossen")

if __name__ == "__main__":
    main()