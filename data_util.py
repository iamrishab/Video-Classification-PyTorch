import os
import cv2
import pandas as pd 
import numpy as np


SUPPORTED_VIDEO_FORMATS = ['mp4', 'MP4', 'AVI', 'avi']

def get_files_dict(data_path):
    data_dict = {} 
    for root, subdir, files in os.walk(data_path):
        for file in [f for f in files if f.split('.')[-1] in SUPPORTED_VIDEO_FORMATS]:
            full_path = os.path.join(root, file)
            parts = file.split('.')
            if parts[0] not in data_dict.keys():
                data_dict[parts[0]] = {}
            data_dict[parts[0]][parts[1]] = full_path
    final_data_dict = {}
    for key in data_dict.keys():
        if len(data_dict[key]) >= 1:
            final_data_dict[key] = data_dict[key]
    return final_data_dict


def get_files_array(data_path):
    data_dict = get_files_dict(data_path)
    data_array = []
    for key in data_dict.keys():
        data_array.append(data_dict[key])
    return data_dict


def create_sequence(data_path):
    data_array = get_files_array(data_path)
    return data_array
