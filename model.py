import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("customers.csv")

model = KMeans(n_clusters=4, random_state=42)
model.fit(data)

data['cluster'] = model.labels_

print(data.head())
