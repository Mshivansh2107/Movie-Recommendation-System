# Movie-Recommendation-System

## Overview
This project is a **Movie Recommendation System** that utilizes **Singular Value Decomposition (SVD)** from the `surprise` library to provide personalized movie recommendations based on user ratings. The system trains on a dataset of movie ratings and predicts ratings for unseen movies, helping users discover films they might enjoy.

## Features
- Loads and preprocesses movie rating data.
- Trains an **SVD-based collaborative filtering model** to predict ratings.
- Generates **personalized recommendations** for users.
- Evaluates the model using **Root Mean Square Error (RMSE)**.
- Visualizes insights such as:
  - **Top 10 most rated movies**
  - **Distribution of ratings**

## Technologies Used
- **Python** (pandas, numpy, matplotlib, surprise)
- **Surprise Library** (for collaborative filtering using SVD)
- **Matplotlib** (for visualizations)
- **Pandas** (for data manipulation)
- **NumPy** (for numerical operations)

## Dataset
The dataset used in this project is sourced from the **MovieLens dataset**, which provides user-movie ratings.
- **Movies Dataset:** Contains movie IDs and titles.
- **Ratings Dataset:** Contains user ratings for movies.

### Dataset Source
The dataset is obtained from **MovieLens**: [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

## Installation
To run this project locally, ensure you have the following dependencies installed:
```bash
pip install pandas numpy surprise matplotlib
```

## Usage
1. Place `movies.csv` and `ratings.csv` in the project directory.
2. Run the script to train the model and get recommendations:
```bash
python movie_recommendation.py
```
3. Modify `user_id` in the script to get recommendations for a specific user.

## Example Output
```
RMSE: 0.87

Top 5 recommendations for user 1:
   movieId          title  estimated_rating
0      100  Movie Title A              4.7
1      200  Movie Title B              4.6
...
```

## Visualization
The script generates two visualizations:
1. **Top 10 Most Rated Movies**
2. **Distribution of Ratings**

## License
This project is for educational purposes only. The MovieLens dataset is provided under its respective license.

## Credits
- **MovieLens Dataset**: [GroupLens Research](https://grouplens.org/datasets/movielens/)
- **Surprise Library**: [https://surpriselib.com](https://surpriselib.com/)


