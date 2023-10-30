import pandas as pd
import numpy as np
import torch


# Create User features
users_data = pd.read_csv('users.csv', usecols=['UserId', 'Gender', 'Age', 'Occupation']) 
gender_map = {"M": 0, "F": 1}

# Apply the mappings to the Gender column
users_data['Gender'] = users_data['Gender'].map(gender_map)

# Perform one-hot encoding for Age and Occupation
one_hot_enc = pd.get_dummies(users_data.drop('UserId', axis=1), columns=['Age', 'Occupation'], dtype=int)

print(f"one_hot_enc")
print(one_hot_enc)
print(one_hot_enc.shape)

# input(" ")
users_data['features']= one_hot_enc.apply(lambda row: row.values, axis=1)
user_features_df= users_data[['UserId','features']]
print('user_features_df.head()')
print(user_features_df.head())


user_features_df.to_csv('user_features.csv', index=False)

# Create the movie features
movies_data = pd.read_csv('movies.csv', usecols=['MovieId', 'Genres'])

# Split the 'Genres' column by '|'
movies_data['Genres'] = movies_data['Genres'].apply(lambda x: x.split('|'))

# Get a list of unique genres
unique_genres = set(genre for genres in movies_data['Genres'] for genre in genres)

# Create one-hot encoded columns for each unique genre
for genre in unique_genres:
    movies_data[genre] = movies_data['Genres'].apply(lambda x: 1 if genre in x else 0)

# Drop the 'Genres' column if necessary
movie_features= movies_data.drop(['Genres','MovieId'], axis=1)

print(f"movie_features.shape: {movie_features.shape}")
# print(movie_features.head())

movies_data['features']= movie_features.apply(lambda row: row.values, axis=1)
movie_features_df= movies_data[['MovieId','features']]
print('movie_features_df.head()')
print(movie_features_df.head())
print(movie_features_df.shape)

movie_features_df.to_csv('movie_features.csv', index=False)

# Load the dataset
data = pd.read_csv('ratings.csv')  # Assuming the file is in the same directory, or specify the path if it's elsewhere

# Find top 100 movies with the most ratings
top_num_movies = 50
top_movies = data['MovieId'].value_counts().head(top_num_movies).index.tolist()

# Create a new dataset with ratings for top_movies
new_data = data[data['MovieId'].isin(top_movies)].copy()

# Calculate the average rating for each movie
avg_ratings = data.groupby('MovieId')['Rating'].mean().reset_index()

# Rename the columns
avg_ratings.columns = ['MovieId', 'AvgRating']

# Check the avg_ratings dataset
print(avg_ratings.head())

avg_ratings.to_csv('avg_ratings.csv', index=False)

# Define a function to fill missing entries for each user
def fill_missing_entries(group, movie_list):
    user_id = group['UserId'].iloc[0]
    rated_movies = set(group['MovieId'])
    missing_movies = list(set(movie_list) - rated_movies)
    num_missing = len(missing_movies)
    max_timestamp = group['Timestamp'].max()
    t_values = np.random.rand(num_missing)
    missing_data = pd.DataFrame({
        'UserId': [user_id] * num_missing,
        'MovieId': missing_movies,
        'Rating': [2.5] * num_missing,
        'Timestamp': max_timestamp + t_values
    })
    return missing_data

# Apply the function to each user group
filled_data = new_data.groupby('UserId').apply(lambda x: fill_missing_entries(x, top_movies)).reset_index(drop=True)


# Concatenate the original data and the filled_data
final_data = pd.concat([new_data, filled_data])

# Convert columns to the integer data type
final_data['UserId'] = final_data['UserId'].astype(int)
final_data['MovieId'] = final_data['MovieId'].astype(int)
final_data['Timestamp'] = final_data['Timestamp'].astype(int)


# Sort the final_data by UserId and Timestamp
final_data.sort_values(['UserId', 'Timestamp'], inplace=True)

# Check final_data
print(final_data.head())

print(f"final_data.shape: {final_data.shape}")

sequence_df = final_data.sort_values(['UserId', 'Timestamp']).groupby('UserId')

print(sequence_df.head())

feature_dim = len(user_features_df['features'].iloc[0]) + len(movie_features_df['features'].iloc[0])

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU")

feature_dim = len(user_features_df['features'].iloc[0]) + len(movie_features_df['features'].iloc[0])

# Create an empty tensor on CUDA device
ml_dataset = torch.zeros((len(sequence_df), top_num_movies, feature_dim), dtype=torch.float32, device=device)

# Get the unique UserIds and MovieIds
unique_user_ids = user_features_df['UserId'].values
unique_movie_ids = movie_features_df['MovieId'].values

# Create a dictionary for easy lookup
user_features_dict = {row['UserId']: row['features'] for _, row in user_features_df.iterrows()}
movie_features_dict = {row['MovieId']: row['features'] for _, row in movie_features_df.iterrows()}

# Iterate through each user's group in sequence_df
for idx, (_, group) in enumerate(sequence_df):
    user_id = group['UserId'].iloc[0]
    user_feature = torch.tensor(user_features_dict[user_id], device=device)

    movie_ids = group['MovieId'].values
    movie_features = [torch.tensor(movie_features_dict[movie_id], device=device) for movie_id in movie_ids]

    concatenated_features = torch.cat((torch.stack([user_feature] * top_num_movies), torch.stack(movie_features)), dim=1)
    ml_dataset[idx, :, :] = concatenated_features

# Move the tensor back to the CPU if necessary
ml_dataset = ml_dataset.cpu() if device == torch.device("cuda") else ml_dataset

# Check the shape of the ML dataset
print("ML dataset shape:", ml_dataset.shape)


# Reshape the ml_dataset tensor into a 2D array
ml_dataset_np = ml_dataset.cpu().numpy() if device == torch.device("cuda") else ml_dataset.numpy()
num_users, top_num_movies, feature_dim = ml_dataset_np.shape
ml_dataset_reshaped = ml_dataset_np.reshape(num_users * top_num_movies, feature_dim)

# Convert the reshaped array to a pandas DataFrame
ml_dataset_df = pd.DataFrame(ml_dataset_reshaped)

# Save the ml_dataset as a CSV file
ml_dataset_df.to_csv('ml_dataset.csv', index=False)

# Create an empty list to store the true_dataset
true_dataset = []

# Iterate through each user's group in sequence_df
for _, group in sequence_df:
    user_id = group['UserId'].iloc[0]
    for _, row in group.iterrows():
        movie_id = row['MovieId']
        avg_rating = avg_ratings[avg_ratings['MovieId'] == movie_id]['AvgRating'].values[0]
        true_dataset.append((user_id, movie_id, avg_rating))

# Convert the true_dataset to a DataFrame
true_dataset_df = pd.DataFrame(true_dataset, columns=['UserId', 'MovieId', 'AvgRating'])

# Save the true_dataset as a CSV file
true_dataset_df.to_csv('true_dataset.csv', index=False)

# Create an empty list to store the user_rating_dataset
user_rating_dataset = []

# Iterate through each user's group in sequence_df
for _, group in sequence_df:
    user_id = group['UserId'].iloc[0]
    for _, row in group.iterrows():
        movie_id = row['MovieId']
        rating = row['Rating']
        user_rating_dataset.append((user_id, movie_id, rating))

# Convert the user_rating_dataset to a DataFrame
user_rating_dataset_df = pd.DataFrame(user_rating_dataset, columns=['UserId', 'MovieId', 'Rating'])

# Save the user_rating_dataset as a CSV file
user_rating_dataset_df.to_csv('user_rating_dataset.csv', index=False)
