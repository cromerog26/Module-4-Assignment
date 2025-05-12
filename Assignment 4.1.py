import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('Spotify_data.csv')

features = ['music_time_slot', 'fav_music_genre', 
            'preferred_listening_content',
            'spotify_subscription_plan', 'music_lis_frequency']

df = pd.get_dummies(df[features])

inertia_values = []
for k in range(2, 61):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(df)
    print(f'k = {k} | inertia = {kmeans.inertia_}')
    inertia_values.append(kmeans.inertia_)

plt.plot(range(2, 61), inertia_values, marker = 'o')
plt.title('Elbow Method for Spotify User Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters = 41)
df['cluster'] = kmeans.fit_predict(df)