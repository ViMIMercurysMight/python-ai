#Task 3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import neighbors, datasets

input_file = 'data.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(np.int_)

plt.figure()
plt.title("InputData")
marker_shapes = 'v^os'
mapper = [marker_shapes[i] for i in y]
for i in range(X.shape[0]):
    plt.scatter(X[i, 0],
                X[i, 1],
                marker=mapper[i],
                s=75,
                edgecolors='black',
                facecolors='none')

num_neighbors = 12
step_size = 0.01

#-----
Classifier = neighbors.KNeighborsClassifier(num_neighbors,
                                            weights='distance')
Classifier.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size),
                                 np.arange(y_min, y_max, step_size))

output = Classifier.predict(np.c_[x_values.ravel(),
                            y_values.ravel()])

output = output.reshape(x_values.shape)
plt.figure()
plt.pcolormesh(x_values, y_values, output, cmap=cm.Paired)

for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1],
                marker=mapper[i],
                s=50,
                edgecolors='black',
                facecolors='none')

plt.xlim(x_values.min(), x_values.max())
plt.ylim(y_values.min(), y_values.max())
plt.title('Edges model classifier on base K nearest neighbors')

test_datapoint = [5.1, 3.6]
plt.figure()
plt.title('Test data point')
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i],
                s=75, edgecolors='black', facecolors='none')

plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', linewidths=6,
            s=200, facecolors='black')

_, indices = Classifier.kneighbors([test_datapoint])
indices = indices.astype(np.int_)[0]

plt.figure()
plt.title('K nearest neighbors')
for i in indices:
    plt.scatter(X[i, 0],
                X[i, 1],
                marker=mapper[y[i]],
                linewidths=3,
                s=100,
                facecolors='black')
plt.scatter(test_datapoint[0], test_datapoint[1], marker='x',
            linewidths=6, s=200, facecolors='black')

for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], s=75,
                edgecolors='black', facecolors='none')

print('Predicted output:', Classifier.predict([test_datapoint])[0])
plt.show()
