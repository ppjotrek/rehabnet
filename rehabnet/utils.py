import scipy
import pandas as pd
import numpy as np
import torch
import torch.utils
import os
import uuid
import json

def load_data(root_dir, target_dir, target_file="data.csv"):
    
    try:
        os.makedirs(target_dir)
    except FileExistsError:
        pass
    
    df = pd.DataFrame(columns=['data', 'label'])
    
    files = os.listdir(root_dir)
    files.sort()
    datafiles = []
    labels = []
    for file in files:
        filename = file.split("_")
        if filename[1] == "data":
            datafiles.append(file)
        elif filename[1] == "labels":
            labels.append(file)

    for i in range(len(datafiles)):
        data = np.load(root_dir + datafiles[i])
        label = np.load(root_dir + labels[i])
        for row in range(data.shape[0]):
            data_item = data[row]
            item_id = str(uuid.uuid4())[:8]
            item_path = target_dir + item_id + ".npy"
            np.save(item_path, data_item)
            temp_df = pd.DataFrame({'data': [item_path], 'label': [label[row]]})
            df = pd.concat([df, temp_df], axis=0)
            
    df.to_csv(target_file, index=False)
    
    class_to_idx_map = {}
    for i in range(len(datafiles)):
        label = np.load(root_dir + labels[i])
        for row in range(label.shape[0]):
            class_to_idx_map[label[row]] = 1
    
    class_to_index_map = json.dumps(class_to_idx_map)
    
    return df, class_to_index_map

class emg_dataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.labels = self.data['label']
        self.data = self.data['data']
        self.class_to_idx_map = {"IDLE": 0, "FD": 1, "HandOpen": 1, "FH": 2, "FL": 3, "FS": 4, "EX": 5, "WR": 6, "WS": 7, "WP": 8, "WU": 9}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.load(self.data[idx])
        data = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        max_data = torch.max(data)
        min_data = torch.min(data)
        data = (data - min_data) / (max_data - min_data)
        data = data.permute(1, 0)
        labels = self.labels[idx]
        labels = self.class_to_idx_map[str(labels)]
        labels = torch.tensor(labels)
        return data, labels