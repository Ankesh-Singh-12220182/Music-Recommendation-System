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
from PIL import Image
import requests
from io import BytesIO

# ------------------------ Load Data ------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("https://drive.google.com/uc?export=download&id=13qnQ6lZq7009GGHIN3AXSMuCkSav4dwo")
    genre_data = pd.read_csv("data_by_genres.csv")
    year_data = pd.read_csv("data_by_year.csv")
    return data, genre_data, year_data

data, genre_data, year_data = load_data()

# ------------------------ Spotify API Setup ------------------------
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="c02e73fdabc14437bd20b5af97e3d461",
    client_secret="774ebcd303104073835c76096b24a1d4"))

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy',
               'explicit', 'instrumentalness', 'key', 'liveness', 'loudness',
               'mode', 'popularity', 'speechiness', 'tempo']

# ------------------------ Clustering ------------------------
@st.cache_resource
def fit_cluster_pipeline():
    X = data.select_dtypes(np.number)
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('kmeans', KMeans(n_clusters=20, n_init=10))])
    pipeline.fit(X)
    return pipeline

song_cluster_pipeline = fit_cluster_pipeline()

# ------------------------ Helper Functions ------------------------

def find_song(name, year):
    song_data = defaultdict()
    try:
        results = sp.search(q=f'track:{name} year:{year}', limit=1)
        if not results['tracks']['items']:
            return None

        result = results['tracks']['items'][0]
        track_id = result['id']
        audio_features = sp.audio_features(track_id)
        if not audio_features or audio_features[0] is None:
            return None
        audio_features = audio_features[0]

        song_data['name'] = [name]
        song_data['year'] = [year]
        song_data['explicit'] = [int(result['explicit'])]
        song_data['duration_ms'] = [result['duration_ms']]
        song_data['popularity'] = [result['popularity']]

        for key, value in audio_features.items():
            song_data[key] = value

        return pd.DataFrame(song_data)

    except spotipy.exceptions.SpotifyException as e:
        st.error(f"Spotify API Error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

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
            st.warning(f"⚠️ {song['name']} ({song['year']}) not found. Skipping.")
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    if not song_vectors:
        return None
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

    if song_center is None:
        return pd.DataFrame(columns=metadata_cols)

    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols]

# ------------------------ Album Image Fetcher ------------------------

DUMMY_IMAGE = "https://via.placeholder.com/300x300.png?text=No+Image"

def get_album_image(song_name, song_year):
    try:
        results = sp.search(q=f'track:{song_name} year:{song_year}', limit=1)
        items = results['tracks']['items']
        if items and 'album' in items[0] and 'images' in items[0]['album'] and items[0]['album']['images']:
            return items[0]['album']['images'][0]['url']
    except:
        pass
    return DUMMY_IMAGE

# ------------------------ Streamlit UI ------------------------

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
        with st.spinner("🔍 Fetching recommendations..."):
            recs = recommend_songs(song_list, data)

        if recs.empty:
            st.warning("No recommendations found. Please check your input songs.")
        else:
            st.subheader("🔁 Recommended Songs:")
            for idx, row in recs.iterrows():
                col1, col2 = st.columns([1, 3])
                with col1:
                    image_url = get_album_image(row['name'], row['year'])
                    st.image(image_url, width=120)
                with col2:
                    st.markdown(f"**🎵 {row['name']}**")
                    st.markdown(f"📅 Year: {row['year']}")
                    if 'artists' in row:
                        st.markdown(f"🎤 Artist(s): {row['artists']}")

# ------------------------ Plots ------------------------

st.markdown("---")
st.subheader("📊 Trend of Musical Features Over the Years")
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(year_data, x='year', y=features, title="Audio Features Over Years")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔥 Top 10 Genres by Popularity")
top10 = genre_data.nlargest(10, 'popularity')
fig2 = px.bar(top10, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
st.plotly_chart(fig2, use_container_width=True)
