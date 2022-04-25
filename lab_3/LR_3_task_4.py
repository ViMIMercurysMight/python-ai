import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Завантажимо набір данних
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Використаємо лише одну "розмірність"
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Розіб'ємо дані на тестові та тренувальні
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Створимо модель лінійної регресії
regr = linear_model.LinearRegression()

# Навчимо
regr.fit(diabetes_X_train, diabetes_y_train)

# Зробимо передбачення
diabetes_y_pred = regr.predict(diabetes_X_test)

# Коефициенти
print("Regression coef: \n", regr.coef_)
print("Regression intercept: \n", regr.intercept_)
# Середня абсолютна похибка
print("Mean absolute error :",
      round(mean_absolute_error(diabetes_y_test, diabetes_y_pred), 2))
# Помилка середньої похибки
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("R2 score: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))


fig, ax = plt.subplots()
ax.scatter(diabetes_y_test, diabetes_y_pred, edgecolors=(0, 0, 0))
ax.plot([diabetes_y.min(), diabetes_y.max()], [diabetes_y.min(), diabetes_y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
