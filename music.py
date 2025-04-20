import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load the datasets
@st.cache_data
def load_data():
    data = pd.read_csv("https://drive.google.com/uc?export=download&id=13qnQ6lZq7009GGHIN3AXSMuCkSav4dwo")
    genre_data = pd.read_csv("data_by_genres.csv")
    year_data = pd.read_csv("data_by_year.csv")
    return data, genre_data, year_data

data, genre_data, year_data = load_data()

# Spotify API Setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="c02e73fdabc14437bd20b5af97e3d461",
    client_secret="774ebcd303104073835c76096b24a1d4"))

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy',
               'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',
               'mode', 'popularity', 'speechiness', 'tempo']

# Clustering model
@st.cache_resource
def fit_cluster_pipeline():
    X = data.select_dtypes(np.number)
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('kmeans', KMeans(n_clusters=20))])
    pipeline.fit(X)
    return pipeline

song_cluster_pipeline = fit_cluster_pipeline()

# Recommendation System Functions
def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q=f'track: {name} year: {year}', limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'].str.lower() == song['name'].lower()) &
                                 (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            st.warning(f"{song['name']} not found.")
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists'] if 'artists' in spotify_data.columns else ['name', 'year']
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)

    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols]

# Streamlit UI
st.title("🎵 Music Recommendation System with Spotify API")

st.sidebar.header("Input Your Favorite Songs")
song_list = []
for i in range(5):
    name = st.sidebar.text_input(f"Song {i+1} Name", key=f"name_{i}")
    year = st.sidebar.number_input(f"Song {i+1} Year", min_value=1900, max_value=2025, step=1, key=f"year_{i}")
    if name and year:
        song_list.append({'name': name, 'year': int(year)})

if st.sidebar.button("Recommend Songs"):
    if len(song_list) == 0:
        st.warning("Please enter at least one song.")
    else:
        with st.spinner("Fetching recommendations..."):
            recs = recommend_songs(song_list, data)
        st.subheader("🔁 Recommended Songs:")
        st.table(recs)

st.markdown("---")
st.subheader("📊 Trend of Musical Features Over the Years")
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(year_data, x='year', y=features, title="Audio Features Over Years")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔥 Top 10 Genres by Popularity")
top10 = genre_data.nlargest(10, 'popularity')
fig2 = px.bar(top10, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
st.plotly_chart(fig2, use_container_width=True)
