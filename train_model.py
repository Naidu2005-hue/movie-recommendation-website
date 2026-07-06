import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Helpers ----------
def convert(obj):
    """Extract 'name' from list of dict strings (genres/keywords)."""
    L = []
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L

def convert_cast(obj):
    """Keep top 3 cast names."""
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i["name"])
            count += 1
        else:
            break
    return L

def fetch_director(obj):
    """Extract director name from crew."""
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            return [i["name"]]
    return []

def clean_list(words):
    """Remove spaces inside names like 'Sam Worthington' -> 'SamWorthington'."""
    return [w.replace(" ", "") for w in words]

# ---------- Load data ----------
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Merge on title (works for TMDB dataset)
movies = movies.merge(credits, on="title")

# Keep needed columns
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies.dropna(inplace=True)

# ---------- Process columns ----------
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(convert_cast)
movies["crew"] = movies["crew"].apply(fetch_director)

movies["overview"] = movies["overview"].apply(lambda x: x.split())

movies["genres"] = movies["genres"].apply(clean_list)
movies["keywords"] = movies["keywords"].apply(clean_list)
movies["cast"] = movies["cast"].apply(clean_list)
movies["crew"] = movies["crew"].apply(clean_list)

# Create tags (combined text features)
movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]

new_df = movies[["movie_id", "title", "tags"]]
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())

# ---------- Vectorize + Similarity ----------
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()

similarity = cosine_similarity(vectors)

# ---------- Save ----------
pickle.dump(new_df, open("model/movies.pkl", "wb"))
pickle.dump(similarity, open("model/similarity.pkl", "wb"))

print("✅ Model saved in model/movies.pkl and model/similarity.pkl")
print("✅ Total movies:", new_df.shape[0])