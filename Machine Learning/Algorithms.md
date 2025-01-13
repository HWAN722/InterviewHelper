# Algorithms

## Supervised Learning
The data set contains input and corresponding correct output, and the goal of the model is to learn the mapping relationship from it so as to be able to predict new unlabeled data. Common supervised learning tasks include *classification* and *regression*.
### Linear Regression
Linear Regression is used for predicting a continuous value. The model assumes a linear relationship between input variables (X) and the single output variable (y).

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 4, 2, 5, 6])

# Model initialization and training
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[6], [7]]))
print(predictions)
```
### Logistic Regression
Logistic Regression is used for binary classification problems. It estimates the probability that an instance belongs to a particular class.
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Model initialization and training
model = LogisticRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(np.array([[6], [7]]))
print(predictions)
```

### SVM, Support Vector Machine
SVM is used for classification or regression tasks. It finds a hyperplane that best separates the classes.

```python
from sklearn import svm
import numpy as np

# Example data
X = np.array([[1, 2], [2, 2], [2, 3], [3, 3]])
y = np.array([0, 0, 1, 1])

# Model initialization and training
model = svm.SVC(kernel='linear')
model.fit(X, y)

# Make predictions
predictions = model.predict([[3, 2]])
print(predictions)
```

### KNN, K-Nearest Neighbors
KNN is a simple, instance-based learning algorithm that classifies instances based on the majority vote of its neighbors.

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Example data
X = np.array([[1, 2], [2, 2], [2, 3], [3, 3]])
y = np.array([0, 0, 1, 1])

# Model initialization and training
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Make predictions
predictions = model.predict([[3, 2]])
print(predictions)
```
### Decision Tree
Decision Trees classify instances by sorting them down the tree from the root node to some leaf node.

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Example data
X = np.array([[1, 2], [2, 2], [2, 3], [3, 3]])
y = np.array([0, 0, 1, 1])

# Model initialization and training
model = DecisionTreeClassifier()
model.fit(X, y)

# Make predictions
predictions = model.predict([[3, 2]])
print(predictions)
```
### Random Forest
Random Forest is an ensemble of decision trees and is used for both classification and regression tasks.

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Example data
X = np.array([[1, 2], [2, 2], [2, 3], [3, 3]])
y = np.array([0, 0, 1, 1])

# Model initialization and training
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# Make predictions
predictions = model.predict([[3, 2]])
print(predictions)
```
### GBM, Gradient Boosting Machine
Gradient Boosting is used for regression and classification. It builds models sequentially, each new model correcting errors made by the previous ones.

```python
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# Example data
X = np.array([[1, 2], [2, 2], [2, 3], [3, 3]])
y = np.array([0, 0, 1, 1])

# Model initialization and training
model = GradientBoostingClassifier(n_estimators=10)
model.fit(X, y)

# Make predictions
predictions = model.predict([[3, 2]])
print(predictions)
```
## Unsupervised Learning
Unsupervised learning does not depend on labeled data, and the model directly discovers the internal structure from the input data. Common tasks include clustering, dimensionality reduction and anomaly detection.

### 1. Clustering
Clustering groups a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups.

### K-Means
K-Means clustering partitions the data into K distinct clusters based on distance to the centroids.

```python
from sklearn.cluster import KMeans
import numpy as np

# Example data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Model initialization and fitting
model = KMeans(n_clusters=2)
model.fit(X)

# Cluster centers
centers = model.cluster_centers_
print("Cluster centers:", centers)

# Predict clusters for new data
predictions = model.predict([[0, 0], [4, 4]])
print(predictions)
```

## Gaussian Mixture Model (GMM)
GMM is a probabilistic model that assumes data is generated from a mixture of several Gaussian distributions.
```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Example data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Model initialization and fitting
model = GaussianMixture(n_components=2)
model.fit(X)

# Predict clusters
labels = model.predict(X)
print("Cluster labels:", labels)
```

### 2. Dimensionality Reduction
Dimensionality Reduction reduces the number of random variables under consideration by obtaining a set of principal variables.

### PCA, Principal Component Analysis
PCA reduces the dimensionality of data while preserving as much variability as possible.

```python
from sklearn.decomposition import PCA
import numpy as np

# Example data
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

# Model initialization and fitting
pca = PCA(n_components=1)
pca.fit(X)

# Transform the data
X_reduced = pca.transform(X)
print("Reduced data:", X_reduced)
```
