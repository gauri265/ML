import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

data=pd.read_csv("customersegment.csv")
X=data[['Annual_Income','Spending_Score']]
dbscan=DBSCAN(eps=5,min_samples=2)
data['Cluster']=dbscan.fit_predict(X)

print(data)

print("\nCluster Labels")
print(data['Cluster'])

print("\nDBSCAN Clustering:")
print(data[['Customer_ID', 'Annual_Income',
            'Spending_Score', 'Cluster']])

plt.scatter(
    X['Annual_Income'],X['Spending_Score'],c=data['Cluster'],s=50)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("DBSCAN Clustering")
plt.show()







