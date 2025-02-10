# CS506-Project

## Music Recommendation Model Proposal

**Project Overview:**
We propose a system that will recommend songs to the user based on the categories of the songs that they have listened to, such as genre and artist. Using data taken from the Spotify API, the application will recommend 15 to 20 songs at a time, as well as display a cluster chart - where the songs are gouped via KNN (K-Nearest Neightbors) - that the user can interact with to see which songs share similarities among each other.

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
To efficiently test and evaluate the recommendation model, we will use a straightforward approach with data splitting, key evaluation metrics, and a basic implementation:
1. **Data Splitting**: we will split the dataset into training and test sets with 80% data is used as training for model learning while 20% data is used as testing for evaluation.
2. **Generating Recommendations**: After the model has been trained, we generate Top-K song recommendations for users using this assumed function **generate_recommendations(user_id, K)** that returns a list of **K recommended songs** for a given user
3. **Precision@K & Recall@K**:
   - Precision@K:
     - Measure how many of the top K recommended songs are relevant
     - Formula: $$Precision@K = \frac{\text{Relevant Items in Top K}}{K}$$
     - E.g. If 2 out of 5 recommended songs are relevant -- Precision@5 = 2/5 = 0.4 (40%)
    - Recall@K:
      - Measure how well the recommendations cover all possible relevant songs
      - Formula: $$Recall@K = \frac{\text{Relevant Recommendations in Top K}}{\text{Total Relevant songs}}$$
      - E.g. If 10 songs are relevant, and the system suggests 2 in the top 5 possible relevent songs -- Recall@5 = 2 / 10 = 0.2 (20%)
     
4. **NDCG@K for Ranking Quality**: NDCG (Normalized Discounted Cumulative Gain) is a metric used to measure the quality of a ranking system. It compares rankings to an ideal order where all relevant items are at the top of the list. In our test, we use NDCG to give higher weight to relevant songs appearing earlier in the recommendation list.
5. **Compute Mean Scores for All Users**: We test our model reliability across multiple users and calculate average scores.
   1. Traversing all user data in the test set
   2. Compute Precision@K, Recall@K, and NCDG@K
   3. Weight and average the results for an overall model evaluation


       
 

