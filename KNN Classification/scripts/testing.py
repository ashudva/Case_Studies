import prediction_grid as pg
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from knn import knn_predict
import matplotlib.pyplot as plt
import numpy as np
import synth_data as sd
# Point to find the distance from
point = np.array([2.4, 2.2])
# Toy data points
points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1],
                   [3, 2], [3, 3]])
# Labels of the correspoinding points
labels = np.array([*np.zeros((4, ), dtype=int), *np.ones((5, ), dtype=int)])

plt.scatter(points[:, 0], points[:, 1], c=labels)
plt.plot(point[0], point[1], "ro", alpha=0.5)
plt.xlim(0.5, 3.5)
plt.ylim(0.5, 3.5)
plt.show()

# Sample generator testing
points, labels = sd.make_synth_data()
plt.scatter(points[:, 0], points[:, 1], c=labels)
plt.show()


points, labels = sd.make_synth_data()
limits = (-3, 4, -3, 4, 0.1)
k = 5
xx, yy, prediction_grid = pg.make_prediction_grid(
    limits,
    points,
    labels,
    k,
)
pg.plot_prediction_grid(xx,
                        yy,
                        prediction_grid,
                        "knn_prediction_grid_5.pdf",
                        points=points,
                        labels=labels,
                        xlab="Variable1",
                        ylab="Variable2",
                        title="Prediction Grid(K=5)")


# Load Iris dataset from sklearn
iris = datasets.load_iris()
predictors = iris['data'][:, :2]
labels = iris['target']
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(predictors[:, 0], predictors[:, 1], c=labels)
pg.annotate(ax,
            xlab=iris['target_names'][0],
            ylab=iris['target_names'][1],
            title="Iris Data Scatter-Plot")
plt.show()
fig.savefig("Iris_scatter.pdf")


limits = (4, 8.2, 1.8, 4.7, 0.1)
xx, yy, prediction_grid = pg.make_prediction_grid(
    limits, predictors, labels, k=5)
pg.plot_prediction_grid(xx,
                        yy,
                        prediction_grid,
                        "Iris_grid_homemade.pdf",
                        points=predictors,
                        labels=labels,
                        xlab=iris['target_names'][0].capitalize(),
                        ylab=iris['target_names'][1].capitalize(),
                        title="Iris dataset prediction-grid using homemade predictor")


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(predictors, labels)
sk_predictions = knn.predict(predictors)
my_predictions = np.array(
    [knn_predict(point, predictors, labels) for point in predictors])
sum(labels == my_predictions) / sk_predictions.shape


limits = (4, 8.2, 1.8, 4.7, 0.1)
xx, yy, prediction_grid = pg.make_prediction_grid_sklearn(
    limits, predictors, labels, k=5)
pg.plot_prediction_grid(xx,
                        yy,
                        prediction_grid,
                        "Iris_grid_sklearn.pdf",
                        points=predictors,
                        labels=labels,
                        xlab=iris['target_names'][0].capitalize(),
                        ylab=iris['target_names'][1].capitalize(),
                        title="Iris dataset prediction-grid Using sklearn")
