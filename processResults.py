import numpy as np
import cv2
import sys
import glob
import copy
import matplotlib.pyplot as plt

BORDER_SIZE_IN_UM=25
SIZE_THRESHOLD_IN_UM2 = 25*25

DEFAULT_PIC_SIZE_IN_PIXEL = 924*1096
DEFAULT_PIXEL_IN_UM = 200/110                #by measurement
DEFAULT_PIC_SIZE_IN_UM2 = DEFAULT_PIC_SIZE_IN_PIXEL * DEFAULT_PIXEL_IN_UM ** 2

BORDER_SIZE_IN_PIXEL = BORDER_SIZE_IN_UM/DEFAULT_PIXEL_IN_UM

VISUALIZATION = False

def contourHitsBorder(contour, shape):
    bsize = int(BORDER_SIZE_IN_UM/DEFAULT_PIXEL_IN_UM * shape[0]*shape[1]/DEFAULT_PIC_SIZE_IN_PIXEL)
    canvas = np.zeros(shape=shape[:2])
    cv2.drawContours(canvas, [contour], -1, 1, -1)
    hit = canvas>0
    return np.any(hit[:bsize]) or np.any(hit[-bsize:]) or np.any(hit[:, :bsize]) or np.any(hit[:, -bsize:])

def contourSizeInUM(contour, shape):
    canvas = np.zeros(shape=shape[:2], dtype=np.uint8)
    cv2.drawContours(canvas, [contour], -1, 1, -1)
    return np.sum(canvas) / shape[0] / shape[1] * DEFAULT_PIC_SIZE_IN_PIXEL*DEFAULT_PIXEL_IN_UM**2

def processResults(contours, masks):
    contours = [[contour for contour in c if not contourHitsBorder(contour, mask.shape)] for c, mask in zip(contours, masks)]

    if VISUALIZATION:
        rects = [[cv2.minAreaRect(oneCont) for oneCont in conts] for conts in contours]
        rectsByPnts = [[cv2.boxPoints(x).astype(int) for x in y] for y in rects]

        i = [copy.deepcopy(m)*128 for m in masks]
        i = [cv2.drawContours(img, cont, -1, 255, 1) for img, cont in zip(i, rectsByPnts)]
        for img in i:
            cv2.imshow("pic", img)
            cv2.waitKey()

    result = dict()
    # 1. avg number of number of blobs
    result["numBlobs"] = [len(contour) for contour in contours]
    blobSizes = [[contourSizeInUM(contour, mask.shape) for contour in cs] for cs, mask in zip(contours, masks)]
    result["avgBlobSize"] =  [np.mean(x) for x in blobSizes]
    result["devBlobSize"] =  [np.std(x) for x in blobSizes]
    result["totalBlobSize"] = [np.sum(x) for x in blobSizes]

    # def giveBigLittleRatio(rectangle):
    #     w, h = rectangle[1]
    #     w, h = (w, h) if w > h else (h, w)
    #     return w / h

    # These are useless with YOLO due to the way it do detection. Luckily, these were bad matrics already.
    # result["avgBlobWpH"] = [np.mean([giveBigLittleRatio(r) for r in rs]) for rs in rects]
    # result["wAvgBlobWpH"] = [np.sum([giveBigLittleRatio(r)*b for r, b in zip(rs, bs)])/np.sum(bs) for rs, bs in zip(rects, blobSizes)]
    return result

if __name__ == "__main__":
    filenames = [filename for fnames in sys.argv[1:] for filename in glob.glob(fnames)]
    filenames.sort()
    masks = [np.load(f"{fname}.mask.npy") for fname in filenames]
    contours = [cv2.findContours(x, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0] for x in masks]

    contours = [[contour for contour in contoursPerPic if contourSizeInUM(contour, mask.shape)>SIZE_THRESHOLD_IN_UM2] for
                contoursPerPic, mask in zip(contours, masks)]

    # rects = [[cv2.minAreaRect(oneCont) for oneCont in conts] for conts in contours]
    # rectsByPnts = [[cv2.boxPoints(x).astype(int) for x in y] for y in rects]

    res = processResults(contours, masks)
    print(res)

    #Number
    plt.plot(np.arange(len(res['numBlobs'])), res['numBlobs'])
    plt.show()

    #Avg Size:
    plt.errorbar(np.arange(len(res['avgBlobSize'])), res['avgBlobSize'], res['devBlobSize'])
    plt.show()

    #Total Size
    plt.plot(np.arange(len(res['totalBlobSize'])), res['totalBlobSize'])
    plt.show()
