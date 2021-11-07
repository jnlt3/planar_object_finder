import cv2 as cv
import numpy as np
import os


DATASET_DIR = "./database/"
QUERIES_DIR = "./queries/"
RESULTS_DIR = "./results/"

RATIO_TEST = 0.7                    # A value in range [0, 1], usually set to 0.7 or 0.8.
NUM_FEATURES = 8192                 # Increasing num features will most likely increase accuracy at the cost of performance
MIN_MATCHES = 20                    # Increasing min matches will cause less false positives (and also less true positives)
ROOT_SIFT = True                    # Root SIFT performs better than SIFT in most cases
BOUNDING_BOX_COLOR = (0, 255, 0)    # BGR e.g. (255, 0, 0) for blue


class KeyPointExtractor:
    def __init__(self):
        self.sift = cv.SIFT_create(nfeatures=NUM_FEATURES)

    def get_keypoints(self, image):
        kp, des = self.sift.detectAndCompute(image, None)
        if ROOT_SIFT:
            des /= (des.sum(axis=1, keepdims=True) + 1e-8)
            des = np.sqrt(des)
        return kp, des


class KeyPointMatcher:

    def __init__(self):
        self.bf_matcher = cv.BFMatcher()

    def match_key_points(self, kp_0, des_0, kp_1, des_1):
        matches_0 = self.bf_matcher.knnMatch(des_0, des_1, k=2)
        matches_1 = self.bf_matcher.knnMatch(des_1, des_0, k=1)
        if matches_0 is None or matches_1 is None:
            return [], []
        matched_queries = []
        matched_trains = []
        for first, second in matches_0:
            ratio_test = first.distance < second.distance * RATIO_TEST
            mutual_test = first.queryIdx == matches_1[first.trainIdx][0].trainIdx
            if mutual_test and ratio_test:
                matched_queries.append(kp_0[first.queryIdx].pt)
                matched_trains.append(kp_1[first.trainIdx].pt)
        return np.array(matched_queries), np.array(matched_trains)


class Solver:
    def solve(self, query, train):
        return cv.findHomography(query, train, cv.USAC_MAGSAC)


KEY_POINT_EXTRACTOR = KeyPointExtractor()
KEY_POINT_MATCHER = KeyPointMatcher()
SOLVER = Solver()


class Dataset:
    def __init__(self, directory: str):
        files = os.listdir(directory)
        self.images = []
        for file_name in files:
            img = cv.imread(directory + file_name)
            if type(img) == np.ndarray:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                m_kp, m_des = KEY_POINT_EXTRACTOR.get_keypoints(img)
                name = file_name[:file_name.rfind('.')]
                self.images.append((m_kp, m_des, img.shape[0], img.shape[1], name))

    def match_images(self, match_img, out_frame):
        found_images = []
        for (kp, des, height, width, name) in self.images:
            if get_matched_image(kp, des, height, width, match_img, out_frame):
                found_images.append(name)
        return found_images


def extract_keypoints(image):
    return KEY_POINT_EXTRACTOR.get_keypoints(image)


def get_matched_image(match_kp, match_des, h, w, frame, out_img):
    kp, des = extract_keypoints(frame)
    if len(kp) < MIN_MATCHES:
        return False
    query, train = KEY_POINT_MATCHER.match_key_points(
        match_kp, match_des, kp, des)
    if len(query) < MIN_MATCHES:
        return False
    M, _ = SOLVER.solve(query, train)
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    cv.polylines(out_img, [np.int32(dst)], True, BOUNDING_BOX_COLOR, 3, cv.LINE_AA)
    return True


def match_and_write(frame, dataset):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return dataset.match_images(frame_gray, frame)


def main():
    dataset = Dataset(DATASET_DIR)

    queries_dir = os.listdir(QUERIES_DIR)

    for file_name in queries_dir:
        file_dir = QUERIES_DIR + file_name
        img = cv.imread(file_dir)
        name = file_name[:file_name.rfind('.')]
        if type(img) == np.ndarray:
            titles = match_and_write(img, dataset)
            titles_file = open(RESULTS_DIR + name + ".txt", "w")
            for title in titles:
                titles_file.write(title + "\n")
            titles_file.close()

            cv.imwrite(RESULTS_DIR + name + ".jpg", img)
        else:
            print("Failed to read file", name)


if __name__ == "__main__":
    main()
