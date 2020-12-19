import knn
import numpy as np
# Prediction Grid


def make_prediction_grid(limits, points, labels, k):
    """Returns coordinates of meshgrid and prediction_grid"""
    x_min, x_max, y_min, y_max, h = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    (xx, yy) = np.meshgrid(xs, ys)
    prediction_grid = np.zeros(xx.shape, dtype=int)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            prediction_grid[j, i] = knn.knn_predict([x, y], points, labels, k)

    return xx, yy, prediction_grid


def annotate(ax, xlab="X", ylab="Y", title="Figure", ticks=False):
    """Annotates a figure, ticks = tuple with xticks and yticks or a boolean"""
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    if not ticks:
        ax.set_xticks(())
        ax.set_yticks(())


def plot_prediction_grid(xx,
                         yy,
                         prediction_grid,
                         filename,
                         points,
                         labels,
                         xlab,
                         ylab,
                         title,
                         ticks=False):
    """ Plot KNN predictions for every point on the grid."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap(
        ["lightpink", "lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap(["red", "blue", "green"])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pcolormesh(xx,
                  yy,
                  prediction_grid,
                  cmap=background_colormap,
                  alpha=0.5,
                  shading="auto",
                  antialiased=True)
    ax.scatter(points[:, 0],
               points[:, 1],
               c=labels,
               cmap=observation_colormap,
               s=50)
    annotate(ax, xlab, ylab, title, ticks)
    ax.set_xlim(np.min(xx), np.max(xx))
    ax.set_ylim(np.min(yy), np.max(yy))
    plt.show()
    fig.savefig(filename)


# Prediction Grid using sklearn knn.predict function
def make_prediction_grid_sklearn(limits, points, labels, k):
    """Returns coordinates of meshgrid and prediction_grid"""
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(points, labels)
    x_min, x_max, y_min, y_max, h = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    (xx, yy) = np.meshgrid(xs, ys)
    prediction_grid = np.zeros(xx.shape, dtype=int)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            prediction_grid[j, i] = knn.predict(
                np.array([x, y]).reshape(1, -1))

    return xx, yy, prediction_grid
