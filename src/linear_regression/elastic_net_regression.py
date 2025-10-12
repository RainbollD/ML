from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, train_test_split

import matplotlib.pyplot as plt

X, y = make_regression(n_samples=1000, n_features=1, noise=5, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

elastic_net = ElasticNet()

param_grid = [
    {'alpha': [0.3, 0.5, 0.7, 1], 'l1_ratio': [0.3, 0.5, 0.7, 1]},
]

grid_search = GridSearchCV(
    estimator=elastic_net,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

plt.scatter(X_train_scaled, y_train, color='red')
plt.scatter(X_test_scaled, y_test, c='blue')
plt.plot(X_train_scaled, grid_search.predict(X_train_scaled))
plt.show()
