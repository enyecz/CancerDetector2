import sys

import numpy as np
import cv2
import copy
import glob

import torch
import tqdm

from models import common
from utils import general
from utils.segment import general as seg_general

DEVICE=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    for x in sys.argv[3:]:
        PFILENAMES = PFILENAMES + glob.glob(x)
    PFILENAMES.sort()
    MFILENAMES = [fname + ".mask.npy" for fname in PFILENAMES]
    print("Loading pics...")
    pics = [cv2.imread(fname)[..., ::-1] for fname in tqdm.tqdm(PFILENAMES)]

    #loading the NN
    model = common.DetectMultiBackend(sys.argv[1], DEVICE, fp16=False)
    inp = [resizeForYolo(pic, int(sys.argv[2])) for pic in pics]
    out = []
    print("Inferencing...")
    for i in tqdm.tqdm(inp):
        tensor = np.transpose(i, (2,0,1))
        tensor = torch.unsqueeze(torch.tensor(tensor).to(DEVICE)/255, 0)
        bbox, preMask = model(tensor)[:2]
        bbox = general.non_max_suppression(bbox, conf_thres=0.4, iou_thres=.2, max_det=1000, nm=32)
        masks = seg_general.process_mask(preMask[0], bbox[0][:, 6:], bbox[0][:, :4], tensor.shape[2:], upsample=True)
        out.append(masks.to('cpu').numpy())
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