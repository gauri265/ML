import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv("customersegment.csv")
X=data[['Annual_Income','Spending_Score']]
kmeans=KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X)

print(data)
print("kmeans cluster centers: ")
print(kmeans.cluster_centers_)
plt.scatter(
    X['Annual_Income'],X['Spending_Score'],c=data['Cluster'],s=50)
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=200,
    marker='X')

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K Means Clustering")
plt.show()


















