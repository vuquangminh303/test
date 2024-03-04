import numpy as np
import cv2
class knnMatcher:
    def __init__(self):
        self.ratio=0.85
        self.min_match=10
        self.smoothing_window_size=800
        pass
    def knnMatch(self,des1,des2,k):
        self.des1=des1
        self.des2=des2
        self.k=k
        matches = []
        good_points = []
        good_matches = []

        for i in range(len(des1)):
            distances = []
            for j in range(len(des2)):
                distance = np.sqrt(np.sum(np.square(np.array(des1[i]) - np.array(des2[j]))))
                distances.append((distance, j))
            
            distances.sort(key=lambda x: x[0])
            matches.append([cv2.DMatch(i, distances[m][1], distances[m][0]) for m in range(k)])

        for m1, m2 in matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])

        return good_points, good_matches

