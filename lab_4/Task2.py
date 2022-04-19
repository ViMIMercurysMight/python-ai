#TASK 2
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import load_iris

iris = load_iris()
X = iris['data']
y = iris['target']

#отримання налаштованого обїекту к середних
kmeans = KMeans(n_clusters=8,
                       init='k-means++',
                       n_init=10,
                       max_iter=300,
                       tol=0.0001,
                       verbose=0,
                       random_state=None,
                       copy_x=True,
                       algorithm='auto'
                       )

#вчимо прочитаними данними
kmeans.fit(X)

#Передбачте найближчий кластер, до якого належить кожна вибірка в X
y_kmeans = kmeans.predict(X)
#Формуємо діаграми розсіювання y та x із різним розміром та кольором маркера.
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

#оголошуемо функцію пошуку кластерів
def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        #Ця функція обчислює для кожного рядка в X індекс рядка Y, який є найближчим (відповідно до вказаної відстані).
        labels = pairwise_distances_argmin(X, centers)
        #обраховуємо середне значення для цетрів
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        print(centers)
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

#знаходимо точки центрів для значення минимальної відстані та встановлюємо точки для діагрмми
centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c = labels, s=50, cmap='viridis')

#Обчислюємо центри кластерів та прогнозуємо індекси кластерів для кожної вибірки
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

plt.show()
