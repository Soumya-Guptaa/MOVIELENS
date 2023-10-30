import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import pickle

seq_len = 50

# Load the ml dataset from the CSV file
ml_dataset_df = pd.read_csv('ml_dataset.csv')

# Extract feature_dim from the column size of entries
feature_dim = len(ml_dataset_df.iloc[0]) 

# Reshape the ml dataset into a tensor
train_tensor = torch.tensor(ml_dataset_df.values).view(-1, seq_len, feature_dim)

# Check the shape of the tensor
print("train_tensor shape:", train_tensor.shape)

# input(" ")

# Load the ml dataset from the CSV file
ratings_df = pd.read_csv('user_rating_dataset.csv',usecols= ['Rating'])
print('ratings_df')
print(ratings_df.head())
print(ratings_df.shape)

# Extract feature_dim from the column size of entries
output_dim = len(ratings_df.iloc[0]) 

# Reshape the ml dataset into a tensor
ratings = torch.tensor(ratings_df.values, dtype=torch.float32).view(-1, seq_len)

# Check the shape of the tensor
# print("ratings shape:", ratings.shape)

# input(" ")


# Load the ml dataset from the CSV file
True_ratings_df = pd.read_csv('true_dataset.csv', usecols= ['AvgRating'])

# Extract feature_dim from the column size of entries
output_dim = len(True_ratings_df.iloc[0]) 

# Reshape the ml dataset into a tensor
True_ratings = torch.tensor(True_ratings_df.values, dtype=torch.float32).view(-1, seq_len)

# Check the shape of the tensor
# print("True_ratings shape:", True_ratings.shape)

# input(" ")

split_size = 0.4
X_train, X_test, y_train, y_test, y_real_train, y_real_test = train_test_split(train_tensor, ratings, True_ratings,
                                                                               test_size=split_size, random_state=42)

print(f"\nX_train.shape: {X_train.shape}, X_test.shape: {X_test.shape},\n y_train.shape: {y_train.shape},  y_test.shape: {y_test.shape},\n y_real_train.shape: {y_real_train.shape}, y_real_test.shape: {y_real_test.shape}")

train_len = len(X_train)
test_len = len(X_test)

# min-max scaling the sequence and the respective scores
# scaler = MinMaxScaler()
# X_train_normalized = scaler.fit_transform(X_train)
# X_test_normalized = scaler.transform(X_test)

# Reshape the 3D arrays to 2D arrays
X_train_2d = X_train.reshape(-1, X_train.shape[-1])
X_test_2d = X_test.reshape(-1, X_test.shape[-1])

# Apply MinMaxScaler across each of the 42 entries for each of the 20 entries separately
scaler = MinMaxScaler()
X_train_normalized_2d = scaler.fit_transform(X_train_2d)
X_test_normalized_2d = scaler.transform(X_test_2d)  # Using the same scaler as in the training set

# Reshape the arrays back to 3D
X_train_normalized = X_train_normalized_2d.reshape(X_train.shape)
X_test_normalized = X_test_normalized_2d.reshape(X_test.shape)

scaler_target = StandardScaler()
y_train_normalized = scaler_target.fit_transform(y_train)
y_test_normalized = scaler_target.transform(y_test)
y_real_train_normalized = scaler_target.transform(y_real_train)

# tensorizing the dataframes
train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)

# original dimension 160*5*1
# train_tensor = dimension_increase(train_tensor) #changing dimension to 160*5*6 by stacking the sequence
test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)

# test_tensor = dimension_increase(test_tensor)
train_score_tensor = torch.tensor(y_train_normalized, dtype=torch.float32)
train_score_tensor = train_score_tensor.unsqueeze(dim=2)
test_score_tensor = torch.tensor(y_test_normalized, dtype=torch.float32)
test_score_tensor = test_score_tensor.unsqueeze(dim=2)
train_real_tensor = torch.tensor(y_real_train_normalized, dtype=torch.float32)
train_real_tensor = torch.unsqueeze(train_real_tensor, dim=2)
test_real_tensor = torch.tensor(y_real_test, dtype=torch.float32)
test_real_tensor = torch.unsqueeze(test_real_tensor, dim=2)

print(f"\ntrain_tensor.shape: {train_tensor.shape}, test_tensor.shape: {test_tensor.shape}\n"
      f"train_score_tensor.shape: {train_score_tensor.shape},test_score_tensor.shape: {test_score_tensor.shape}\n"
      f"train_real_tensor.shape: {train_real_tensor.shape}, test_real_tensor.shape: {test_real_tensor.shape}\n")


def preprocess_data():
    # Save train and test data to separate files
    with open(f"train_tensor_{int((1 - split_size) * 100)}_{int(split_size * 100)}.pkl", 'wb') as f:
        pickle.dump(train_tensor, f)

    with open(f"train_score_tensor_{int((1 - split_size) * 100)}_{int(split_size * 100)}.pkl", 'wb') as f:
        pickle.dump(train_score_tensor, f)

    with open(f"train_real_tensor_{int((1 - split_size) * 100)}_{int(split_size * 100)}.pkl", 'wb') as f:
        pickle.dump(train_real_tensor, f)

    with open(f"test_tensor_{int((1 - split_size) * 100)}_{int(split_size * 100)}.pkl", 'wb') as f:
        pickle.dump(test_tensor, f)

    with open(f"test_score_tensor_{int((1 - split_size) * 100)}_{int(split_size * 100)}.pkl", 'wb') as f:
        pickle.dump(test_score_tensor, f)

    with open(f"test_real_tensor_{int((1 - split_size) * 100)}_{int(split_size * 100)}.pkl", 'wb') as f:
        pickle.dump(test_real_tensor, f)

    with open('scaler_target.pkl', 'wb') as f:
        pickle.dump(scaler_target, f)

    return train_tensor, train_score_tensor, train_real_tensor, test_tensor, test_score_tensor, test_real_tensor, scaler_target
