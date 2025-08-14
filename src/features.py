# vo_project/features.py

import cv2

class FeatureExtractor:
    """Abstract base class for feature extractors."""
    def extract(self, image):
        raise NotImplementedError

class ORBExtractor(FeatureExtractor):
    """ORB feature extractor."""
    def __init__(self, n_features=3000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
    
    def extract(self, image):
        return self.orb.detectAndCompute(image, None)

class FeatureMatcher:
    """Abstract base class for feature matchers."""
    def match(self, des1, des2):
        raise NotImplementedError

class BruteForceMatcher(FeatureMatcher):
    """Brute-force feature matcher with Lowe's ratio test."""
    def __init__(self, norm_type=cv2.NORM_HAMMING):
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)

    def match(self, des1, des2):
        if des1 is None or des2 is None:
            return []
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            # This can happen if k=2 is not possible (e.g., not enough features)
            pass
        return good_matches