# src/data_loader.py
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

def load_ratings(path='../data/u.data'):
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='\t', names=column_names)
    df = df.dropna() 
    return df

def load_movies(path='../data/u.item'):
    movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + \
                    [f'genre_{i}' for i in range(19)]
    df = pd.read_csv(path, sep='|', names=movie_columns, encoding='latin-1')
    genre_columns = [col for col in df.columns if col.startswith('genre_')]
    return df[['movie_id', 'title'] + genre_columns]

def load_users(path='../data/u.user'):
    user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    df = pd.read_csv(path, sep='|', names=user_columns)
    df = df.dropna() 
    return df

def prepare_surprise_dataset(ratings_df):
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    return trainset, testset