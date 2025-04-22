# -Movie-Watch-Pattern-Clustering_202401100300151
# 🎬 Movie Watch Pattern Clustering

This project performs **unsupervised clustering** of users based on their movie-watching behavior, including the **time of watching**, **genre preference**, and **average rating behavior**. The goal is to group similar users to understand viewing habits, which can be useful for personalization and recommendation systems.

---

## 📁 Dataset

The dataset used is `movie_watch.csv` and contains the following columns:

- `watch_time_hour`: Hour of the day the movie was watched (0-23)
- `genre_preference`: Preferred movie genre (e.g., Action, Comedy)
- `avg_rating_given`: Average rating given by the user (1.0 - 5.0)

---

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Seaborn & Matplotlib

---

## 🧾 Objective

To cluster users based on:
- The time they watch movies (`watch_time_hour`)
- Their favorite genre (`genre_preference`)
- The average rating they give (`avg_rating_given`)

We use **KMeans clustering** to identify patterns and **PCA** to visualize the results.

---

## 🔍 How the Code Works

### 1. Import Libraries
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("movie_watch.csv")
label_encoder = LabelEncoder()
df['genre_encoded'] = label_encoder.fit_transform(df['genre_preference'])
features = df[['watch_time_hour', 'genre_encoded', 'avg_rating_given']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10')
plt.title('Movie Watch Pattern Clustering (PCA View)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.tight_layout()
plt.show()



📊 Output
A 2D scatter plot of users colored by cluster.

Shows how people group by their movie habits (e.g., night-action fans, morning-comedy lovers, etc.).

💻 How to Run
Clone the repo:
git clone https://github.com/your-username/movie-clustering.git
cd movie-clustering
Install dependencies:
pip install pandas scikit-learn matplotlib seaborn
Run the script:
python cluster_movie_watch.py


