# --- Machine Learning Libraries ---
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import config

class MovieLensDataset(Dataset):
    def __init__(self, df, categorical_features):
        self.data = {}
        for feature in config.categorical_features:
            self.data[feature] = torch.LongTensor(df[feature].values)
        self.targets = torch.FloatTensor(df['target'].values)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        sample = {feature: self.data[feature][idx] for feature in self.data}
        return sample, self.targets[idx]


if __name__ == "__main__":

    # load data
    train_df = pd.read_csv('dataprep/train.csv')
    test_df = pd.read_csv('dataprep/test.csv')

    print(f"Dataset shape: {train_df.shape}")
    print(f"Target distribution: {train_df['target'].value_counts()}")


    # Feature dimensions
    feature_dims = {
        'UserID': train_df['UserID'].nunique(),
        'MovieID': train_df['MovieID'].nunique(),
        'Gender': train_df['Gender'].nunique(),
        'Age': train_df['Age'].nunique(),
        'Occupation': train_df['Occupation'].nunique()
    }

    # Create datasets and dataloaders
    train_dataset = MovieLensDataset(train_df, config.categorical_features)
    test_dataset = MovieLensDataset(test_df, config.categorical_features)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # 1. Convert the DataLoader into a Python Iterator
    data_iter = iter(train_loader)

    # 2. Extract exactly one batch of data
    batch = next(data_iter)

    # 3. If your DataLoader returns features and labels: (images, labels) or (inputs, targets)
    inputs, targets = batch

    # 4. Print shapes and data types
    print("--- Single Batch Inspection ---")
    print(f"Inputs Shape : {len(inputs.keys()), len(inputs['UserID'])}")   
    # print(f"Inputs Type  : {inputs[0].dtype}")   # e.g., torch.float32
    print(f"Targets Shape: {targets.shape}")  # e.g., torch.Size([32])
    print(f"Targets Type : {targets.dtype}")  # e.g., torch.int64