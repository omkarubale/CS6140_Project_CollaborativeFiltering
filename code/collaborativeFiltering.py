import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader, SVD
from surprise.model_selection import cross_validate

# Setting up movie data
df1 = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

df1.columns = ['id','title','cast','crew']

# print(df2.head(5))

# Setting up rating data
reader = Reader()
ratings = pd.read_csv('../input/the-movies-dataset/ratings_small.csv')

# print(ratings.head())
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# data.split(n_folds=5)

# Calculating RMSE and MAE of algorithm SVD
svd = SVD()
# TODO: make own test-train split 
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train Dataset and prepare to predict
trainset = data.build_full_trainset()
svd.fit(trainset)

print("Ratings: ", ratings[ratings['userId'] == 1])

print("SVD Predictions: ", svd.predict(1, 302, 3).est)

# TODO: calc ndcg
# ndcg needs 2 arrays, one with expected rankings, one with actual rankings

# step 1: test train split per user (80% of movie rating in training set, 20% in test)
# step 2: train model (svd fit)
# step 3: iterate: test model using test split of data, use RMSE for metric
# step 4: group by user (use pandas.group_by), calc NDCG for each user, take avg of NDCG over all users (top 5 for NDCG)