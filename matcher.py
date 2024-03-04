import numpy as np
from scipy.spatial import distance

class KNN:
    def __init__(self):
        pass
    def knnMatch(self,des1,des2,k):
        self.des1=des1
        self.des2=des2
        self.k=k
        raw_matches = []
        for i in range(len(self.des1)):
            distances = [distance.euclidean(self.des1[i], des2_j) for des2_j in self.des2]
            idx = np.argsort(distances)
            matches = [(i, idx[j], distances[idx[j]]) for j in range(self.k)]
            raw_matches.append(matches)
        return raw_matches
