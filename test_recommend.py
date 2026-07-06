import pickle

movies = pickle.load(open("model/movies.pkl", "rb"))
similarity = pickle.load(open("model/similarity.pkl", "rb"))

def recommend(movie):
    movie = movie.lower()
    matches = movies[movies["title"].str.lower() == movie]
    if matches.empty:
        return ["Movie not found 😢"]

    idx = matches.index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    recs = []
    for i in distances[1:11]:
        recs.append(movies.iloc[i[0]].title)
    return recs

print(recommend("Avatar"))