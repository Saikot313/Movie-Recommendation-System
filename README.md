# Movie Recommendation System

A machine learning project that implements three recommendation approaches — Content-Based Filtering, Collaborative Filtering, and a Hybrid system — using the MovieLens dataset.

---

##  Dataset

| Property | Value |
|----------|-------|
| Source | [MovieLens](https://grouplens.org/datasets/movielens/) |
| Movies | 10,329 |
| Ratings | 105,339 |
| Users | 668 |
| Rating Scale | 0.5 – 5.0 |

**Required files:** `movies.csv`, `ratings.csv`

---

##  Techniques Used

| Approach | Method | Best For |
|----------|--------|----------|
| Content-Based Filtering | TF-IDF + Cosine Similarity | New users, genre-based discovery |
| Collaborative Filtering | SVD + KNN (User-Item Matrix) | Users with rating history |
| Hybrid System | CB (40%) + CF (60%) | Best overall accuracy |

---

##  Project Structure

```
Movie_Recommendation_System/
│
├── Movie_Recommendation_System.ipynb   # Main notebook
├── movies.csv                          # Movie titles and genres
├── ratings.csv                         # User ratings
└── README.md
```

To get recommendations for any movie:

```python
# Content-Based
content_based_recommend("Inception", n=10)

# Collaborative Filtering
collaborative_recommend("Inception", n=10)

# Hybrid
hybrid_recommend("Inception", n=10)

# By genre
search_by_genre("Action", min_ratings=30, n=10)
```

---

##  How It Works

### Content-Based Filtering
Genres are vectorized using **TF-IDF** and movie similarity is measured via **Cosine Similarity**. Recommends movies with the closest genre profile to the input title.

### Collaborative Filtering
Builds a **User-Item matrix** (top 2,000 movies by rating count). **TruncatedSVD** reduces dimensionality to 50 components, then **KNN with cosine distance** finds similar movies based on user rating patterns.

### Hybrid System
Combines both scores with configurable weights (default: CB=0.4, CF=0.6). Falls back to Content-Based if a movie lacks sufficient CF data.

---

##  Model Evaluation

The SVD-based Collaborative Filtering is evaluated using **RMSE** on an 80/20 train-test split. Lower RMSE indicates better rating prediction accuracy on the 0.5–5.0 scale.

---

##  Dependencies

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | TF-IDF, SVD, KNN, metrics |
| `matplotlib`, `seaborn` | Visualization |

---

##  Author

**Saikot**  
MSc in Data Science — AIUB  
GitHub: [@Saikot313](https://github.com/Saikot313)
