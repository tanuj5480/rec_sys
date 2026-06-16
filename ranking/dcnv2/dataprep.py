# --- Core Libraries ---
import os
import zipfile
import requests
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
import config



# --- Machine Learning Libraries ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Environment Setup ---
os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# --- 1. DATA PREPARATION ---
print("--- Step 1: Preparing MovieLens 1M Data ---")

# Download and extract data
# if not os.path.exists('ml-1m.zip'):
#     print("Downloading MovieLens 1M dataset...")
#     os.system("wget -q http://files.grouplens.org/datasets/movielens/ml-1m.zip")
#     os.system("unzip -q ml-1m.zip")

# Load data
data_dir = "ml-1m/"
ratings_df = pd.read_csv(os.path.join(data_dir, "ratings.dat"), sep='::', engine='python', 
                        names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
users_df = pd.read_csv(os.path.join(data_dir, "users.dat"), sep='::', engine='python',
                      names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
movies_df = pd.read_csv(os.path.join(data_dir, "movies.dat"), sep='::', engine='python',
                       names=['MovieID', 'Title', 'Genres'], encoding='latin-1')

# Merge data
df = ratings_df.merge(users_df, on='UserID').merge(movies_df, on='MovieID')

# Create binary target (1 if rating >= 4, 0 otherwise)
df['target'] = (df['Rating'] >= 4).astype(int)

# Encode categorical features
encoders = {}

for feature in config.categorical_features:
    encoders[feature] = LabelEncoder()
    df[feature] = encoders[feature].fit_transform(df[feature])

print(f"Dataset shape: {df.shape}")
print(f"Target distribution: {df['target'].value_counts()}")

# Split data

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

# df.to_csv('dataprep/final_data.csv', index=False)
train_df.to_csv('dataprep/train.csv', index=False)
test_df.to_csv('dataprep/test.csv', index=False)