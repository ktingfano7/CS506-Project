# CS506-Project

## Music Recommendation Model Proposal (TBD)

**Project Overview:**

**Goals:**
The goal of this project is to build an interactive music recommendation system using K-Nearest Neighbors (KNN). The system will allow users to select songs and visually see how the algorithm finds the most similar songs based on Spotify audio features.

**Data Collection:**
We will use Spotify’s Web API to fetch song metadata and audio features. The API provides detailed track analysis, which we can extract using Python.

**Model Selection & Training:**
Since we are recommending songs based on similarity, we will use K-Nearest Neighbors (KNN) to find the closest songs based on audio features.
What will the model do:
- Preprocessing: it will normalize the features for consistency
- Train KNN model on Euclidean distance or Cosine similarity to measure similarity.
- When a user selects a song, find the K most similar songs. (We will determine k later)
- Show a step-by-step visualization of how the KNN model finds similar songs.

**Data Visualization:**
We will create an interactive visualization that dynamically demonstrates how a song is found using K-Nearest Neighbors (KNN).  

**Test & Evaluation Plan:**


