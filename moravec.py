import cv2
import numpy as np
import time


# r and c represent the row and colon, while u and v represent the shift
def shift_np(src, r, c, u, v):
    return ((src[r-1:r+2, c-1:c+2] - src[r+u-1:r+u+2, c+v-1:c+v+2])**2).sum()


def calculate_intensity_variation(src):
    intensity_variation = np.zeros((src.shape[0], src.shape[1], 8), np.int32)
    shifts = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

    for row in range(2, src.shape[0]-2):
        for col in range(2, src.shape[1]-2):
            for i in range(8):
                intensity_variation[row, col, i] = shift_np(src, row, col, *shifts[i])

    return intensity_variation


def calculate_cornerness_measure(intensity_variation, t=20_000):
    cornerness_measure = np.zeros((intensity_variation.shape[0], intensity_variation.shape[1]), np.int32)
    for row in range(0, intensity_variation.shape[0]):
        for col in range(0, intensity_variation.shape[1]):
            minimum = np.min(intensity_variation[row, col])
            cornerness_measure.itemset((row, col), minimum if minimum >= t else 0)
    return cornerness_measure


def non_maximum_suppression(src):
    maximized = np.zeros(src.shape, np.uint8)
    for row in range(1, src.shape[0]-1):
        for col in range(1, src.shape[1]-1):
            if src[row, col] == 0:
                continue

            local_maxima = True
            u = -1; v = -1
            while local_maxima and u <= 1:
                while local_maxima and v <= 1:
                    if u == 0 and v == 0:
                        continue
                    elif src.item((row, col)) < src.item((row + u, col + v)):
                        local_maxima = False
                        break
                    v += 1
                u += 1
            if local_maxima:
                maximized.itemset((row, col), 255)
    return maximized


# Algorithm idea :
# https://arxiv.org/ftp/arxiv/papers/1209/1209.1558.pdf

filename = "resources/star.jpg"; threshold = 25_000
# filename = "resources/lena.png"; threshold = 7_000
# filename = "resources/chessboard.jpg"; threshold = 20_000
# filename = "resources/inclined-chessboard.jpg"; threshold = 20_000

img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("int32")

print("Calculating intensity variation")
start_time = time.time()
variation = calculate_intensity_variation(gray)
end_time = time.time()
print("Elapsed time :", end_time-start_time, "s", end='\n')

# Uncomment to help choose a threshold.
# print("Variation max", variation.max())

print("Calculating cornerness measure and thresholding")
start_time = time.time()
cornerness = calculate_cornerness_measure(variation, threshold)
end_time = time.time()
print("Elapsed time :", end_time-start_time, "s", end='\n')

print("Operating non maximum suppression")
start_time = time.time()
cornerness = non_maximum_suppression(cornerness)
end_time = time.time()
print("Elapsed time :", end_time-start_time, "s", end='\n')

# Drawing a circle for each corner.
rows, cols = np.where(cornerness != 0)
for idx in range(len(rows)):
    cv2.circle(img, (cols[idx], rows[idx]), 5, (0, 0, 255))

cv2.imshow("Moravec", img)
cv2.waitKey()
