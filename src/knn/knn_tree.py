from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
print(f1_score(y_test, predictions))