from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

# Authenticate
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

# Replace with your folder IDs
FOLDER_IDS = {
    "DermMel": "1KN_r_JkTpnnc3Tp9ngUf_f70snebJChE",
    # Add more datasets here
}

def download_folder(folder_id, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    for file in file_list:
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            # Recursively download subfolders
            download_folder(file['id'], os.path.join(dest_path, file['title']))
        else:
            print(f"Downloading {file['title']} -> {dest_path}")
            file.GetContentFile(os.path.join(dest_path, file['title']))

for dataset_name, folder_id in FOLDER_IDS.items():
    print(f"Downloading dataset: {dataset_name}")
    download_folder(folder_id, os.path.join("datasets", dataset_name))
