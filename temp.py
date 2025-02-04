# Import necessary libraries
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt

# Load and prepare the data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Initialize the Reader object with the rating scale
reader = Reader(rating_scale=(1, 5))

# Load the dataset into the format required by scikit-surprise
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize the SVD model for collaborative filtering
model = SVD()

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Calculate RMSE for evaluation
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Function to get movie recommendations for a specific user
def get_movie_recommendations(user_id, num_recommendations=5):
    # Get a list of all movie IDs
    movie_ids = ratings['movieId'].unique()
    # Filter out movies the user has already rated
    rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId']
    movies_to_predict = [movie_id for movie_id in movie_ids if movie_id not in rated_movie_ids.values]

    # Predict ratings for each unrated movie
    predictions = [model.predict(user_id, movie_id) for movie_id in movies_to_predict]
    # Sort predictions by estimated rating, descending
    predictions.sort(key=lambda x: x.est, reverse=True)
    # Get the top recommendations
    top_predictions = predictions[:num_recommendations]
    # Fetch movie titles for the top recommendations
    recommended_movie_ids = [pred.iid for pred in top_predictions]
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    # Create a DataFrame for the recommended movies
    recommendations = pd.DataFrame({
        'movieId': recommended_movies['movieId'].values,
        'title': recommended_movies['title'].values,
        'estimated_rating': [pred.est for pred in top_predictions]
    })
    return recommendations

# Example: Get recommendations for a specific user (e.g., user with ID 1)
user_id = 1
num_recommendations = 5
recommendations = get_movie_recommendations(user_id, num_recommendations)
print(f"\nTop {num_recommendations} recommendations for user {user_id}:")
print(recommendations)

# Plot the top 10 most rated movies based on the number of ratings
top_10_most_rated = ratings.groupby('movieId').size().sort_values(ascending=False).head(10)
top_10_most_rated_movies = movies[movies['movieId'].isin(top_10_most_rated.index)]
top_10_most_rated_movies['num_ratings'] = top_10_most_rated.values

# Plot the top 10 most rated movies
plt.figure(figsize=(10, 6))
plt.barh(top_10_most_rated_movies['title'], top_10_most_rated_movies['num_ratings'], color='lightgreen')
plt.xlabel('Number of Ratings')
plt.title('Top 10 Most Rated Movies')
plt.gca().invert_yaxis()  # Invert the y-axis to have the most rated movie at the top
plt.show()

# Plot the distribution of ratings (1 to 5)
plt.figure(figsize=(10, 6))
plt.hist(ratings['rating'], bins=np.arange(0.5, 6, 1), edgecolor='black', color='skyblue')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.xticks(np.arange(1, 6, 1))  # Set the x-ticks to the rating scale (1-5)
plt.show()
