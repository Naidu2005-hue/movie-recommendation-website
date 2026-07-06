import pandas as pd

movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

print("Movies:", movies.shape)
print("Credits:", credits.shape)
print(movies["title"].head())