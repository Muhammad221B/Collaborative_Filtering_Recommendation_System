from surprise import KNNBasic

def train_user_based_cf(trainset):
    """
    Train a User-Based Collaborative Filtering model using Surprise.
    
    Args:
        trainset: Surprise trainset object.
    
    Returns:
        algo: Trained User-Based CF model.
    """
    sim_options = {
        'name': 'cosine', 
        'user_based': True  
    }
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    return algo

def train_item_based_cf(trainset):
    """
    Train an Item-Based Collaborative Filtering model using Surprise.
    
    Args:
        trainset: Surprise trainset object.
    
    Returns:
        algo: Trained Item-Based CF model.
    """
    sim_options = {
        'name': 'cosine',
        'user_based': False  
    }
    algo = KNNBasic(sim_options=sim_options)
    algo.fit(trainset)
    return algo

def get_top_n_recommendations(algo, user_id, movies_df, ratings_df, n=5):
    """
    Get Top-N movie recommendations for a user.
    
    Args:
        algo: Trained Surprise model (User-Based or Item-Based).
        user_id: ID of the user to get recommendations for.
        movies_df: DataFrame containing movie information (movie_id, title, etc.).
        ratings_df: DataFrame containing user ratings (user_id, item_id, rating).
        n (int): Number of recommendations to return.
    
    Returns:
        list: List of tuples (movie_id, title, predicted_rating) for top-N recommendations.
    """
    all_movie_ids = movies_df['movie_id'].unique()
    
    rated_movies = ratings_df[ratings_df['user_id'] == user_id]['item_id'].values
    
    unrated_movies = [mid for mid in all_movie_ids if mid not in rated_movies]
    
    predictions = [algo.predict(user_id, mid) for mid in unrated_movies]
    
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_n = predictions[:n]
    
    top_n_movies = [(pred.iid, movies_df[movies_df['movie_id'] == pred.iid]['title'].values[0], pred.est) 
                    for pred in top_n]
    return top_n_movies