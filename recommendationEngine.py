#app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("imdb_top_1000.csv")

# Clean and preprocess
features = ['Genre', 'IMDB_Rating', 'Meta_score', 'Director']
for feature in features:
    df[feature] = df[feature].fillna('')
df['IMDB_Rating'] = df['IMDB_Rating'].astype(str)
df['Meta_score'] = df['Meta_score'].astype(str)
df['index'] = df.index
df['combined_features'] = df.apply(
    lambda row: row['Genre'] + " " + row['IMDB_Rating'] + " " + row['Meta_score'] + " " + row['Director'], axis=1
)

# Vectorize
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

# Get movie recommendations
def recommend_movies(movie_title, top_n=5):
    if movie_title not in df['Title'].values:
        return []

    movie_index = df[df['Title'] == movie_title]['index'].values[0]
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    recommendations = []
    for i, score in sorted_similar_movies:
        recommendations.append({
            "title": df.iloc[i]["Title"],
            "poster": df.iloc[i]["Poster_Link"]
        })
    return recommendations

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

movie_list = df["Title"].tolist()
selected_movie = st.selectbox("Select a movie you like:", movie_list)

if st.button("Get Recommendations"):
    results = recommend_movies(selected_movie)
    if results:
        st.subheader("Top Recommendations:")
        cols = st.columns(len(results))
        for idx, col in enumerate(cols):
            with col:
                st.image(results[idx]["poster"], use_column_width=True)
                st.caption(results[idx]["title"])
    else:
        st.warning("No recommendations found. Please try another movie.")
