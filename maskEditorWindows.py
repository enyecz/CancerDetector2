import sys

import cv2
import numpy as np
import copy
import glob
import tqdm

COLORS = [(0,0,0), (255,0,0), (0,255,0), (0,0,255), (255,255,0),
(0,255,255), (255,0,255), (192,192,192), (128,128,128), (128,0,0),
(128,128,0), (0,128,0), (128,0,128), (0,128,128), (0,0,128)]

FILENAMES = []
for x in sys.argv[1:]:
    FILENAMES = FILENAMES + glob.glob(x)
FILENAMES.sort()
print("Loading images...")
pics  = [cv2.imread(fname) for fname in tqdm.tqdm(FILENAMES)]
masks = []

print("Loading/creating masks...")
for pic, fname in tqdm.tqdm(zip(pics, FILENAMES), total=len(pics)):
    try:
        masks.append(np.load(fname + ".mask.npy"))
    except:
        masks.append(np.zeros(shape=pic.shape[:2], dtype=np.uint8))


class EditorWindow:         #This is a singleton class
    OFF = 0
    DELETE = 1
    DRAW = 2
    GRAB = 3

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):        # We have one instance
            cls.instance = super(EditorWindow, cls).__new__(cls) # if we already have, return the address of the instance
        return cls.instance

    def __init__(self, categories):
        if not hasattr(EditorWindow, 'initialized'):   # initialize the instance only once
            self.CIRC_SIZE = 10
            self.ZOOM = 1.0
            self.START = [0, 0]
            self.prev = [0, 0]
            self.windowName = "Editor"
            self.image = None
            self.mask = None
            self.state = EditorWindow.OFF
            self.editType = 1
            self.circleCenter = None
            EditorWindow.initialized = True
            cv2.namedWindow(self.windowName, cv2.WINDOW_GUI_NORMAL)
            cv2.setMouseCallback(self.windowName, EditorWindow.mouseCallback)
            self.categories = categories

    def setImage(self, image, mask = None):
        self.image = (image//2).astype(np.uint8)
        if mask.shape[0] == image.shape[0] and mask.shape[1] == image.shape[1]:
            self.mask = mask
        else:
            raise Exception("Mask dimensions are not the same. Resize is not implemented yet.")
        self.calculateImage()

    def calculateImage(self):
        if self.image is None:
            return
        else:
            maskPart = np.take(COLORS, self.mask, axis=0)
            self.processedMask = copy.deepcopy(self.image)
            self.processedMask[self.mask>0] = (maskPart*0.5)[self.mask>0]
            self.processedMask = self.processedMask.astype(np.uint8)

    def updateImage(self):
        if self.image is not None and self.processedMask is not None:
            curr = self.image + self.processedMask

            for cnt in range(len(self.categories)):
                cv2.line(curr, (20, 15 + cnt * 10), (curr.shape[1] // 12 - 20, 15 + cnt * 10), COLORS[cnt + 1],
                         thickness=5)
                cv2.putText(curr, "{} - {}".format(self.categories[cnt], cnt), (curr.shape[1] // 12, 20 + cnt * 10),
                            cv2.FONT_HERSHEY_SIMPLEX, curr.shape[0] / 1200, color=(0, 0, 0))
            if self.circleCenter is not None:
                curr = cv2.circle(curr, self.circleCenter, round(self.CIRC_SIZE), (0, 0, 0), thickness=1)
            x = cv2.resize(curr, (0, 0), fx=self.ZOOM, fy=self.ZOOM)
            x = x[self.START[0]:self.image.shape[0] + self.START[0], self.START[1]:self.image.shape[1] + self.START[1]]
            cv2.imshow(self.windowName, x)

    def showImage(self):
        self.calculateImage()
        self.updateImage()

    def getMask(self):
        return self.mask

    @staticmethod
    def mouseCallback(event, xa, ya, flags, param):
        this = EditorWindow.instance       #the singleton object

        x = int(this.START[1]/this.ZOOM + xa/this.ZOOM)
        y = int(this.START[0]/this.ZOOM + ya/this.ZOOM)

        if event == cv2.EVENT_MOUSEWHEEL:
            if flags>0:
                this.START[0] += round(0.1 / this.ZOOM*(ya + this.START[0]))
                this.START[1] += round(0.1 / this.ZOOM*(xa + this.START[1]))
                this.ZOOM += 0.1
                this.START[0] = min(this.START[0], round((this.ZOOM - 1) * EditorWindow.instance.image.shape[0]))
                this.START[1] = min(this.START[1], round((this.ZOOM - 1) * EditorWindow.instance.image.shape[1]))
            else:
                this.START[0] -= int(0.1 / this.ZOOM*(ya + this.START[0]))
                this.START[1] -= int(0.1 / this.ZOOM*(xa + this.START[1]))
                this.ZOOM -= 0.1
                this.START[0] = max(0, this.START[0])
                this.START[1] = max(0, this.START[1])
            this.ZOOM = np.max([this.ZOOM, 1.0])

        if event == cv2.EVENT_RBUTTONDOWN:
            this.state = EditorWindow.DELETE

        if event == cv2.EVENT_LBUTTONDOWN:
            this.state = EditorWindow.DRAW

        if event == cv2.EVENT_MBUTTONDOWN:
            this.state = EditorWindow.GRAB

        if event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_MBUTTONUP:
            this.state = EditorWindow.OFF

        if this.state == EditorWindow.DELETE:
            cv2.circle(this.mask, (x,y), round(np.round(this.CIRC_SIZE)), 0, thickness=-1)
            this.calculateImage()

        if this.state == EditorWindow.DRAW:
            cv2.circle(this.mask, (x,y), round(this.CIRC_SIZE), this.editType, thickness=-1)
            c = (np.array(COLORS[this.editType])/2).astype(np.uint8).tolist()
            cv2.circle(this.processedMask, (x, y), round(this.CIRC_SIZE), c, thickness=-1)

        if this.state == EditorWindow.GRAB:
            xo, yo = this.prev
            this.START[1] -= x-xo
            this.START[0] -= y-yo
            this.START[0] = min(max(this.START[0], 0), round((this.ZOOM - 1) * this.image.shape[0]))
            this.START[1] = min(max(this.START[1], 0), round((this.ZOOM - 1) * this.image.shape[1]))

        this.circleCenter = (x,y)
        this.updateImage()

    def save(self, maskFileName):
        np.save(maskFileName,  self.mask)

editor = EditorWindow(["Cell"])

pos = 0
maxPos = len(pics)-1
while True:
    EditorWindow.instance.setImage(pics[pos], masks[pos])
    EditorWindow.instance.showImage()
    key = cv2.waitKey() & 0xFF
    if key == ord("n"):
        if pos < maxPos:
            masks[pos] = EditorWindow.instance.mask
            EditorWindow.instance.save(f"{FILENAMES[pos]}.mask.npy")
            pos += 1

    if key == ord("p"):
        if pos > 0:
            masks[pos] = EditorWindow.instance.mask
            EditorWindow.instance.save(f"{FILENAMES[pos]}.mask.npy")
            pos -= 1

    if key == ord("s"):
        EditorWindow.instance.save(f"{FILENAMES[pos]}.mask.npy")

    if key == ord("+"):
        EditorWindow.instance.CIRC_SIZE += 1

    if key == ord("-"):
        if EditorWindow.instance.CIRC_SIZE >1:
            EditorWindow.instance.CIRC_SIZE -=1


    if key == ord("q"):
        break

