"""
Read our custom-collected json files
"""
import os
from glob import glob
import json
import numpy as np
from torch.utils.data import Dataset


def delete_superfluous_files():
    path = f'/mnt/hdd/Gait'
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.startswith('._.DS_Store') or name.startswith('._gait_') or name.startswith('.DS_Store'):
                file_path = os.path.join(root, name)
                os.remove(file_path)
                print(f'Deleted: {file_path}')


def read_json():
    path = f'/mnt/hdd/Gait/gait_dataset_keypoints/output_json_folder'

    # for root, dirs, files in os.walk(path):
    #     for name in files:
    #         if name.endswith('.json'):
    #             file_path = os.path.join(root, name)
    #             print(file_path)
    #             with open(file_path, 'r') as f:
    #                 data = json.load(f)
    #                 print(data)

    pattern = f'/user_*/session_*/*/'
    matching_folders = glob(path + pattern, recursive=True)
    print("Matching folders:")
    for folder in matching_folders:
        print(folder)  # split by matching_folders
        # TODO: each 'output_json_folder/{user_id}/{session_id}/{activity_instances[ai]}/{camera_view}/' 
        # is a numpy array/pandas df, ref Hamza's code
    

if __name__ == '__main__':
    # delete_superfluous_files()
    read_json()
