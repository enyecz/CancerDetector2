import numpy as np
import cv2
import sys

DEFAULT_RES = [1096, 924]
DEFAULT_PIC_SIZE_IN_PIXEL = np.prod(DEFAULT_RES)
DEFAULT_PIXEL_IN_UM = 200/110                #by measurement

SIZE_THRESHOLD_IN_UM2 = 10*10

def contourSizeInUM(contour, shape):
    canvas = np.zeros(shape=shape[:2], dtype=np.uint8)
    cv2.drawContours(canvas, [contour], -1, 1, -1)
    return np.sum(canvas) / shape[0] / shape[1] * DEFAULT_PIC_SIZE_IN_PIXEL*DEFAULT_PIXEL_IN_UM**2

if __name__ == "__main__":
    mask = np.load(sys.argv[1])
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [contour for contour in contours if contourSizeInUM(contour, mask.shape)>SIZE_THRESHOLD_IN_UM2]

    if len(sys.argv)>=4:
        img = cv2.imread(sys.argv[3])
        img = cv2.drawContours(img, contours, -1, (255,0,0), 2)
        cv2.imshow("VIS", img)
        cv2.waitKey()

    with open(sys.argv[2], 'wt') as f:
        for contour in contours:
            nextStr = "0"
            for dat in (contour/[[DEFAULT_RES]]).flatten():
                nextStr = f"{nextStr} {dat}"

            f.write(f"{nextStr}\n")

    print("ready")