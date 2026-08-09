"""Read and display an image supplied on the command line."""

import argparse
from pathlib import Path

import cv2


parser = argparse.ArgumentParser()
parser.add_argument("image", type=Path)
args = parser.parse_args()

image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(args.image)
cv2.imshow("image", image)
cv2.waitKey()
