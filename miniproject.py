import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

unlabelled = pd.read_csv("./dataset/social-media.csv")
labelled = pd.read_csv("./dataset/social-media-labelled.csv")

scaler = StandardScaler()
X_unlabelled = scaler.fit_transform(unlabelled)

kmeans = KMeans(n_clusters=len(labelled['label'].unique()), random_state=42)
clusters = kmeans.fit_predict(X_unlabelled)
unlabelled['cluster'] = clusters

X_labelled = scaler.transform(labelled.drop('label', axis=1))
y_labelled = labelled['label']

clf = LogisticRegression()
clf.fit(X_labelled, y_labelled)

predicted_labels = clf.predict(X_unlabelled)
unlabelled['predicted_label'] = predicted_labels

unlabelled.to_csv("./dataset/unlabelled_labelled.csv", index=False)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_unlabelled)

plt.figure(figsize=(12,10))
plt.scatter(X_tsne[:,0], X_tsne[:,1], 
            c=unlabelled['cluster'], cmap='viridis', 
            s=60, alpha=0.8, edgecolors='k')
plt.title("KMeans Clusters (t-SNE Projection)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label="Cluster")
plt.show()

plt.figure(figsize=(12,10))
plt.scatter(X_tsne[:,0], X_tsne[:,1], 
            c=unlabelled['predicted_label'], cmap='plasma', 
            s=60, alpha=0.8, edgecolors='k')
plt.title("Predicted Labels (Logistic Regression - t-SNE Projection)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.colorbar(label="Predicted Label")
plt.show() 