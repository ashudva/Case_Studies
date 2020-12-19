import numpy as np
import scipy.stats as ss
import random


def distance(p1, p2):
    """Returns the distance between two points p1 and p2"""
    return np.sqrt(np.sum((p1 - p2)**2))


def majority_vote(votes):
    """Returns majority vote and its count in a list of votes"""
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    winners = []
    max_counts = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_counts:
            winners.append(vote)
    return random.choice(winners)


def majority_vote_short(votes):
    """Returns majority vote and its count in a list of votes"""
    mode, count = ss.mstats.mode(votes)
    return mode, count


def knn_predict(point, points, labels, k=5):
    """Returns the predicted label of the point"""
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(point, points[i])
    ind = np.argsort(distances)
    return majority_vote(labels[ind][:k])
