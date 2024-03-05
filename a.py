import cv2
import numpy as np
class ImageStiching:
    def __init__(self,ratio=0.75,min_match=10,smoothing_window_size=800):
        self.ratio=ratio
        self.min_match=min_match
        self.smoothing_window_size=smoothing_window_size
        self.sift=cv2.SIFT_create()
    def searchkp(self,img1,img2):
        kp1,des1=self.sift.detectAndCompute(img1,None)
        kp2,des2=self.sift.detectAndCompute(img2,None)
        matcher=cv2.BFMatcher()
        raw_matcher=matcher.knnMatch(des1,des2,k=2)
        good_points=[]
        good_matches=[]
        for m1,m2 in raw_matcher:
            if m1.distance< self.ratio*m2.distance:
                good_points.append((m1.trainIdx,m1.queryIdx))
                good_matches.append(m1)
        good_matches=sorted(good_matches,key= lambda x: x.distance,reverse=True)
        good_matches=good_matches[:200]
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite('matching.jpg', img3)
        if len(good_points)>self.min_match:
            pts1=np.float32([kp1[i].pt for (_,i) in good_points])
            pts2=np.float32([kp2[i].pt for (i,_) in good_points])
        H,status=cv2.findHomography(pts1,pts2,cv2.RANSAC)
        return H
    def ghep(self,img1,img2):
        H=self.searchkp(img1,img2)
        h1,w1=img1.shape[:2]
        h2,w2=img2.shape[:2]
        res=cv2.warpPerspective(img1,H,(w1+w2,h1))
        res[0:h2,0:w2]=img2
        return res
img1=cv2.imread('h3.jpg')
img2=cv2.imread('h4.jpg')
res=ImageStiching().ghep(img1,img2)
cv2.imwrite('res1.jpg',res)
cv2.waitKey(0)