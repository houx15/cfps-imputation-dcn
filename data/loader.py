import pandas as pd
import numpy as np
from utils import LOGGER
from typing import Dict, Tuple
import torch
import os

# 构造torch的dataset
from torch.utils.data import Dataset, DataLoader

# encode
from sklearn.preprocessing import LabelEncoder


def load_data(raw_path: str, embedding_path: str) -> Tuple[pd.DataFrame, Dict]:
    temp_root = "data/temp"
    if not os.path.exists(temp_root):
        os.makedirs(temp_root)

    raw_data_temp_path = os.path.join(temp_root, "raw_data_temp.pkl")
    embedding_temp_path = os.path.join(temp_root, "embedding_temp.pkl")

    if os.path.exists(raw_data_temp_path) and os.path.exists(embedding_temp_path):
        LOGGER.info(f"Loading data from {raw_data_temp_path} and {embedding_temp_path}")
        raw_data = pd.read_pickle(raw_data_temp_path)
        embedding_data = pd.read_pickle(embedding_temp_path)
        LOGGER.info(f"Data loaded")
        return raw_data, embedding_data

    LOGGER.info(f"Loading data from {raw_path} and {embedding_path}")
    raw_data = pd.read_parquet(raw_path)
    embedding_data = np.load(embedding_path, allow_pickle=True).item()
    # 对variable列进行label encode
    le = LabelEncoder()
    le.fit(list(embedding_data.keys()))
    raw_data["variable"] = le.transform(raw_data["variable"])
    le_pid = LabelEncoder()
    le_pid.fit(raw_data["pid"])
    raw_data["pid"] = le_pid.transform(raw_data["pid"])
    le_year = LabelEncoder()
    le_year.fit(raw_data["year"])
    raw_data["year"] = le_year.transform(raw_data["year"])
    # raw_data drop na按行
    raw_data = raw_data.dropna()

    new_embedding_data = {}
    # 变tensor
    for key in embedding_data:
        embedding_data[key] = torch.tensor(embedding_data[key], dtype=torch.float32)
        new_embedding_data[le.transform([key])[0]] = embedding_data[key]
    LOGGER.info(f"Data loaded")

    # save
    raw_data.to_pickle(raw_data_temp_path)
    # save dict
    pd.to_pickle(new_embedding_data, embedding_temp_path)
    return raw_data, new_embedding_data


class DCNDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.pid = torch.tensor(data["pid"].values, dtype=torch.float32)
        self.year = torch.tensor(data["year"].values, dtype=torch.float32)
        self.variable = torch.tensor(data["variable"].values, dtype=torch.float32)
        self.X = torch.stack([self.pid, self.year, self.variable], dim=1)
        self.y = torch.tensor(data["answer"].values, dtype=torch.float32)
        self.y = self.y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_data_loader(
    data: pd.DataFrame, embedding: Dict, batch_size: int, num_worker: int
) -> DataLoader:
    dataset = DCNDataset(data)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker
    )
