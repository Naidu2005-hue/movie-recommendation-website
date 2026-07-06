from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load saved model files
movies = pickle.load(open("model/movies.pkl", "rb"))
similarity = pickle.load(open("model/similarity.pkl", "rb"))

movie_list = movies["title"].tolist()


def recommend(movie_name):
    # Case insensitive match
    matches = movies[movies["title"].str.lower() == movie_name.lower()]

    if matches.empty:
        return []

    idx = matches.index[0]

    distances = sorted(
        list(enumerate(similarity[idx])),
        reverse=True,
        key=lambda x: x[1]
    )

    recommended_movies = []

    for i in distances[1:11]:  # Top 10 recommendations
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    error = None
    selected_movie = ""

    if request.method == "POST":
        selected_movie = request.form.get("movie").strip()
        recommendations = recommend(selected_movie)

        if not recommendations:
            error = "Movie not found! Please type correct movie name."

    return render_template(
        "index.html",
        movies=movie_list,
        recommendations=recommendations,
        error=error,
        selected_movie=selected_movie
    )


if __name__ == "__main__":
    app.run(debug=True)