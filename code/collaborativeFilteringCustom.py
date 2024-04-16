from collections import defaultdict
import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import ndcg_score


# Setting up movie data
df1 = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

df1.columns = ["id", "title", "cast", "crew"]

# print(df2.head(5))

# Setting up rating data

ratings = pd.read_csv("../input/the-movies-dataset/ratings_small.csv")

# group ratings by user for test-train split by user
user_ratings = defaultdict(list)
user_ids = []
for _, row in ratings.iterrows():
    user_ratings[row["userId"]].append(
        (row["userId"], row["movieId"], row["timestamp"], row["rating"])
    )
    user_ids.append(row["userId"])

# Split data into testset and trainset
train_ratio = 0.8
train_ratings = []
test_ratings = []
for user_id, ratings in user_ratings.items():
    n_ratings = len(ratings)
    n_train = int(train_ratio * n_ratings)
    train_ratings.extend(ratings[:n_train])
    test_ratings.extend(ratings[n_train:])

train_user_df = pd.DataFrame(
    train_ratings, columns=["userId", "movieId", "timestamp", "rating"]
).drop(["timestamp"], axis=1)
test_user_df = pd.DataFrame(
    test_ratings, columns=["userId", "movieId", "timestamp", "rating"]
).drop(["timestamp"], axis=1)

# Shuffle Train/Test DataFrames
train_user_df = train_user_df.sample(frac=1).reset_index(drop=True)
test_user_df = test_user_df.sample(frac=1).reset_index(drop=True)

# prepare trainset and testset in required format
reader = Reader()
trainset_data = Dataset.load_from_df(
    train_user_df[["userId", "movieId", "rating"]], reader
)
trainset = trainset_data.build_full_trainset()

testset_list = test_user_df.values.tolist()

# region HELPER: check test train split per user

# m = {"train": {}, "test": {}}
# print("trainset: ", trainset.all_ratings())

# for userId, movieId, rating in trainset.all_ratings():
#     if userId in m["train"]:
#         m["train"][userId] = m["train"][userId] + 1
#     else:
#         m["train"][userId] = 1

# for userId, movieId, rating in testset:
#     if userId in m["test"]:
#         m["test"][userId] = m["test"][userId] + 1
#     else:
#         m["test"][userId] = 1

# print("m[\"train\"]: ", m["train"][0])
# print("m[\"test\"]: ", m["test"][0])
# print("testset: ", testset_list)

# endregion

# train Dataset and prepare to predict
algo = SVD()
algo.fit(trainset)

# region HELPER: confirmation that SVD worked

# validate using NDCG
# print("Ratings for User 1: ", train_user_df[train_user_df['userId'] == 1])
# print("SVD Predictions: ", algo.predict(1, 302, 3))

# endregion

# TODO: calc ndcg
ndcg_scores = []

# for each user:
# ndcg needs 2 arrays: one with expected rankings, one with actual rankings
for user_id in user_ids:
    ndcg_actual_ratings = []
    ndcg_predicted_ratings = []
    print(test_user_df[
        test_user_df["userId"] == user_id
    ])

    for index, row in test_user_df[
        test_user_df["userId"] == user_id
    ].iterrows():
        # print(row)
        user_id, movie_id, actual_rating = row
        predicted_rating = algo.predict(user_id, movie_id, actual_rating).est
        print(f"User {user_id} actual rating: {actual_rating}")
        print(f"User {user_id} predicted rating: {predicted_rating}")
        ndcg_actual_ratings.append(actual_rating)
        ndcg_predicted_ratings.append(predicted_rating)

    user_ndcg_score = ndcg_score(ndcg_actual_ratings, ndcg_predicted_ratings)
    print(f"User {user_id} ndcg score: {user_ndcg_score}")
    ndcg_scores.append(user_ndcg_score)

print(ndcg_scores)


def Average(lst):
    return sum(lst) / len(lst)


print(f"Avg NDCG Score: ${Average(ndcg_scores)}")
