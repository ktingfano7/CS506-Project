import streamlit as st
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

def normalize_string(s):
    return re.sub(r"\W+", " ", s).strip().lower()

def process_track_name(name):
    return normalize_string(re.sub(r'[-\(].*?[\)]', '', name))

def process_artist(artist):
    return ';'.join(sorted(normalize_string(a) for a in artist.split(';')))

@st.cache_data
# Load and preprocess data once
def load_data(path='dataset.csv'):
    df = pd.read_csv(path)
    # clean names
    df['clean_track']  = df['track_name'].apply(process_track_name)
    df['clean_artist'] = df['artists'].apply(process_artist)
    # encode genre_code
    if not pd.api.types.is_integer_dtype(df['genre_code']):
        le = LabelEncoder()
        df['genre_code'] = le.fit_transform(df['genre_code'])
    # scale features
    features = ['popularity','danceability','energy','loudness',
                'speechiness','acousticness','instrumentalness',
                'liveness','valence','tempo']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    return df, scaled

# Return list of recommendation dicts
def get_recommendations(df, scaled, track, artist, metric, n_songs, same_genre):
    # build KNN model
    n_neighbors = len(df) if same_genre else n_songs+1
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(scaled)
    # find query index
    clean_t = process_track_name(track)
    clean_a = process_artist(artist)
    mask = (df['clean_track']==clean_t) & (df['clean_artist']==clean_a)
    if not mask.any():
        return None, f"Track '{track}' by {artist} not found"
    idx = df[mask].index[0]
    dists, idxs = knn.kneighbors([scaled[idx]])
    # collect results
    recs = []
    for dist, i in zip(dists[0], idxs[0]):
        if i == idx:
            continue
        if same_genre and df.loc[i, 'genre_code'] != df.loc[idx, 'genre_code']:
            continue
        score = (1-dist) if metric=='cosine' else dist
        recs.append({
            'track': df.loc[i,'track_name'],
            'artist': df.loc[i,'artists'],
            'score': round(score, 3)
        })
        if len(recs) >= n_songs:
            break
    return recs, None

# ---------- Streamlit App ----------
def main():
    st.title("🎵 KNN Song Recommender")
    # load data
    df, scaled = load_data()
    # UI controls
    artist = st.selectbox("Choose your favorite artist", sorted(df['artists'].unique()))
    tracks = sorted(df[df['artists']==artist]['track_name'].unique())
    track = st.selectbox("Choose a track", tracks)
    metric = st.selectbox("Distance metric", ['cosine','euclidean','manhattan'])
    n_songs = st.slider("How many recommendations?", 1, 20, 5)
    same_genre = st.checkbox("Restrict to same genre", value=False)
    # trigger
    if st.button("Get Recommendations"):
        recs, error = get_recommendations(df, scaled, track, artist,
                                           metric, n_songs, same_genre)
        if error:
            st.error(error)
        else:
            # display results as a table
            st.table(pd.DataFrame(recs))

if __name__=='__main__':
    main()
