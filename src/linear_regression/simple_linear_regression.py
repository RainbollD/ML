import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, random_state=100, noise=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(mse)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.7, label='Реальные значения')
plt.scatter(X_test, y_pred, alpha=0.7, label='Предсказания')
plt.plot(X_test, model.predict(X_test), color='red', linewidth=2, label='Линия регрессии')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Линейная регрессия без регуляризации')
plt.show()