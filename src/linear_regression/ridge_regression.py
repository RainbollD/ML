from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000,
    n_features=2,
    noise=1,
    random_state=123
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ridge = Ridge()

params_grim = {
    'alpha': [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4],
}

grid_search = GridSearchCV(estimator=ridge, param_grid=params_grim, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
