#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

def normalize_string(s):
    return re.sub(r'\W+', ' ', s).strip().lower()

def process_track_name(name):
    # remove parenthetical parts, lowercase, strip punctuation
    return normalize_string(re.sub(r'[-\(].*?[\)]', '', name))

def process_artist(artist):
    # split on ';', normalize each, sort (to handle multi‐artist consistently)
    return ';'.join(sorted(normalize_string(a) for a in artist.split(';')))

def find_similar_songs(df, scaled_features, metric, song_name, artist_name, n_results):
    # fit a KNN model with just enough neighbors
    knn = NearestNeighbors(n_neighbors=n_results+1, metric=metric)
    knn.fit(scaled_features)
    
    clean_track = process_track_name(song_name)
    clean_artist = process_artist(artist_name)
    mask = (
        (df['clean_track'] == clean_track) &
        (df['clean_artist'] == clean_artist)
    )
    if not mask.any():
        print(f"Error: '{song_name}' by {artist_name} not found in database.")
        return
    
    idx = df[mask].index[0]
    dists, idxs = knn.kneighbors([scaled_features[idx]], n_neighbors=n_results+1)
    
    results = []
    for dist, i in zip(dists[0], idxs[0]):
        if i == idx:
            continue
        track = df.loc[i, 'track_name']
        artist = df.loc[i, 'artists']
        # for cosine, smaller distance = more similar, so we convert to “similarity”
        score = (1 - dist) if metric == 'cosine' else dist
        results.append((track, artist, score))
        if len(results) >= n_results:
            break
    
    print(f"\nTop {len(results)} similar songs to '{song_name}' ({metric}):")
    for rank, (t, a, s) in enumerate(results, 1):
        print(f"{rank}. {t} — {a} | Score: {s:.3f}")

def find_similar_songs_genre(df, scaled_features, metric, song_name, artist_name, n_results):
    # make sure genre_code is numeric
    if df['genre_code'].dtype != np.integer:
        le = LabelEncoder()
        df['genre_code'] = le.fit_transform(df['genre_code'])
    
    knn = NearestNeighbors(n_neighbors=len(df), metric=metric)
    knn.fit(scaled_features)
    
    mask = (df['track_name']==song_name) & (df['artists']==artist_name)
    if not mask.any():
        print(f"Error: '{song_name}' by {artist_name} not found in database.")
        return
    
    idx = df[mask].index[0]
    target_genre = df.loc[idx, 'genre_code']
    dists, idxs = knn.kneighbors([scaled_features[idx]], n_neighbors=len(df))
    
    results = []
    for dist, i in zip(dists[0], idxs[0]):
        if i == idx:
            continue
        if df.loc[i, 'genre_code'] != target_genre:
            continue
        track = df.loc[i, 'track_name']
        artist = df.loc[i, 'artists']
        score = (1 - dist) if metric == 'cosine' else dist
        results.append((track, artist, score))
        if len(results) >= n_results:
            break
    
    print(f"\nTop {len(results)} similar songs to '{song_name}' in same genre ({metric}):")
    for rank, (t, a, s) in enumerate(results, 1):
        print(f"{rank}. {t} — {a} | Score: {s:.3f}")

def main():
    parser = argparse.ArgumentParser(
        description="Find similar songs using a KNN-based audio‐feature model"
    )
    parser.add_argument('--dataset', default='dataset.csv',
                        help='path to your CSV of tracks and features')
    parser.add_argument('--song',    required=True,
                        help='exact track name to query')
    parser.add_argument('--artist',  required=True,
                        help='exact artist name to query')
    parser.add_argument('--metric', choices=['cosine','euclidean','manhattan'],
                        default='cosine', help='distance metric for KNN')
    parser.add_argument('--n_songs', type=int, default=5,
                        help='how many similar songs to return')
    parser.add_argument('--same_genre', action='store_true',
                        help='restrict results to the same genre_code')
    args = parser.parse_args()
    
    # load and preprocess
    df = pd.read_csv(args.dataset)
    df['clean_track']  = df['track_name'].apply(process_track_name)
    df['clean_artist'] = df['artists'].apply(process_artist)
    
    features = ['popularity','danceability','energy','loudness',
                'speechiness','acousticness','instrumentalness',
                'liveness','valence','tempo']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    
    # choose mode
    if args.same_genre:
        find_similar_songs_genre(df, scaled,
                                 args.metric,
                                 args.song, args.artist,
                                 args.n_songs)
    else:
        find_similar_songs(df, scaled,
                           args.metric,
                           args.song, args.artist,
                           args.n_songs)

if __name__ == '__main__':
    main()
