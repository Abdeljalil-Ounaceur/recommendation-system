import gradio as gr
import numpy as np
import pandas as pd
import pickle

# Define the prediction function (copying from previous notebook cell)
def predict_rating(user_index, movie_index, user_embeddings, item_embeddings):
    """Predicts the rating for a given user and movie using their embeddings."""
    return np.dot(user_embeddings[user_index], item_embeddings[movie_index])

# --- Code for loading the model and data ---
# Load embeddings and mappings
user_embeddings = np.load('user_embeddings.npy')
item_embeddings = np.load('item_embeddings.npy')
with open('user_to_index.pkl', 'rb') as f:
    user_to_index = pickle.load(f)
with open('movie_to_index.pkl', 'rb') as f:
    movie_to_index = pickle.load(f)

# Load movie data for titles
names=movie_columns = [
    "movie_id", "title", "release_date", "video_release_date",
    "IMDb_URL", "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_clean = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", names=names).drop(columns=["unknown", "video_release_date", "IMDb_URL"])


# --- Code for the recommendation function will go here ---
# def recommend_movies(user_id, top_n=10):
    # Implementation using loaded embeddings and mappings


# --- Code for creating the Gradio interface will go here ---
# iface = gr.Interface(...)
# iface.launch()

# --- Code for the recommendation function ---
def recommend_movies(user_id, top_n=10):
    """
    Generates top-N movie recommendations for a given user ID.

    Args:
        user_id (int): The ID of the user for whom to generate recommendations.
        top_n (int, optional): The number of top recommendations to return.
                                Defaults to 10.

    Returns:
        list: A list of strings, where each string is a recommendation
              formatted as "Movie Title: Predicted Rating". Returns an error
               message if the user ID is not found.
    """
    # Get the index for the user
    user_index = user_to_index.get(user_id)

    if user_index is None:
        return [f"Error: User ID {user_id} not found in the dataset."]

    # Get the list of all movie IDs
    all_movie_ids = movies_clean['movie_id'].unique()

    # Predict ratings for all movies for the given user
    predicted_ratings_all = []
    for movie_id in all_movie_ids:
        movie_index = movie_to_index.get(movie_id)
        if movie_index is not None:
            predicted_rating = predict_rating(user_index, movie_index, user_embeddings, item_embeddings)
            predicted_ratings_all.append((movie_id, predicted_rating))

    # Sort the movies by predicted rating in descending order
    predicted_ratings_all.sort(key=lambda x: x[1], reverse=True)

    # Get the top N recommendations
    top_recommendations = predicted_ratings_all[:top_n]

    # Format the output
    formatted_recommendations = []
    for movie_id, predicted_rating in top_recommendations:
        # Get the movie title
        movie_title = movies_clean[movies_clean['movie_id'] == movie_id]['title'].iloc[0]
        formatted_recommendations.append(f"{movie_title}: {predicted_rating:.4f}")

    return formatted_recommendations

# --- Code for creating the Gradio interface will go here ---
# iface = gr.Interface(...)
# iface.launch()

# --- Code for creating the Gradio interface ---
iface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Number(label="Enter User ID"),
    outputs=gr.Textbox(label="Recommended Movies", interactive=False),
    title="Movie Recommender",
    description="Enter a user ID to get movie recommendations."
)

iface.launch(debug=True, share=True)
