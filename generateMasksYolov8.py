import sys

import numpy as np
import cv2
import copy
import glob
import tqdm
import torch

import ultralytics

def resizeForYolo(pic, imgsz):
    bigger, smaller  = np.max(pic.shape[:2]), np.min(pic.shape[:2])
    smallerFinal = int(np.ceil(imgsz/bigger*smaller/32)*32)
    if pic.shape[0] > pic.shape[1]:
        return cv2.resize(pic, (smallerFinal, imgsz))
    else:
        return cv2.resize(pic, (imgsz, smallerFinal))


if __name__ == "__main__":
    #loading the pics
    PFILENAMES = []
    for x in sys.argv[2:]:
        PFILENAMES = PFILENAMES + glob.glob(x)
    PFILENAMES.sort()
    MFILENAMES = [fname + ".mask.npy" for fname in PFILENAMES]
    print("Loading pics...")
    pics = [cv2.imread(fname)[..., ::-1] for fname in tqdm.tqdm(PFILENAMES)]

    #loading the NN
    yolo = ultralytics.YOLO(sys.argv[1])
    inp = [resizeForYolo(pic, yolo.overrides['imgsz']) for pic in pics]

    # There is a bug/feature that yolov8 eats BGR images for predict.
    # However, since yolo otherwise use RGB everywhere, I use this trick to avoid
    # problems, if they find out that this is a bug and fix it in the future.
    # So now, we neutralize the conversion and our yolo definitely use RGB.
    yolo.predict(inp[0], imgsz=512, verbose=False)
    yolo.predictor.model.model.transforms = lambda im: np.ascontiguousarray(im.transpose((2, 0, 1)))

    out = []
    print("Inferencing...")
    for i in tqdm.tqdm(inp):
        tmp = yolo.predict(i, imgsz=512, iou=.2, conf=0.4, verbose=False)
        out.append(tmp[0].masks.masks.to('cpu').numpy())
    # for n in range(0, len(inp), BATCH):
    #     tmp = model.predict(inp[n*BATCH:(n+1)*BATCH], iou=.2, conf=0.4, batch=4)
    #     out.extend([x.masks.masks.to('cpu').numpy() for x in tmp])

    combinedMasks = []
    for masksPerPic, pic in zip(out, pics):
        thisCombMask = np.zeros(shape=out[0].shape[1:], dtype=np.uint8)
        for mask in masksPerPic:
            thisCombMask[mask>0] = 255    # we have just one class, but we will do a resize!
        thisCombMask = cv2.resize(thisCombMask, (pic.shape[1], pic.shape[0]), interpolation=cv2.INTER_LINEAR)
        thisCombMask[thisCombMask<128] = 0
        thisCombMask[thisCombMask>=128] = 1
        combinedMasks.append(thisCombMask)

    print("Saving resulting masks...")
    for mask, maskName in tqdm.tqdm(zip(combinedMasks, MFILENAMES), total=len(combinedMasks)):
        np.save(maskName, mask)

    print("Creating visualization images...")
    for pic, mask, name in tqdm.tqdm(zip(pics, combinedMasks, PFILENAMES), total=len(pics)):
        tmp = copy.deepcopy(pic)
        pic[mask>0] = pic[mask>0]//2 + [[[128, 0, 0]]]
        cv2.imwrite(name + ".vis.png", pic[..., ::-1])
        # cv2.imshow("Visualization", pic)
        # cv2.waitKey()