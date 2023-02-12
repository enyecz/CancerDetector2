## <div align="center">CancerBlob Detector</div>

Note: directory utils and models are from project [YOLOv5](
https://github.com/ultralytics/yolov5) and not part of this project.
They are here to make it simpler to use the code.

### 1 What is this?

This is a special research project to study the behavior of 
cancer cells in the presence of various poisonous molecules.
In some special cases, cancer cells form coherent blobs in order
to defend themselves from the toxic agent. This repo is a machine
vision solution aiming to automatically detect such blobs in
microscope images.

Note that this is an AS IS code, no help, no warranty, it may kill
your kitten.

### 2 Prerequests

The code is written in Python 3. Tested with Python 3.9 and 3.10. If training
is needed it is strongly recommend to have one/some GPU(s) with
PyTorch support. Tested with Nvidia GTX 1060, RTX 2080Ti and
RTX 3090, but should work perfectly with any other GPU with PyTorch
support enabled.

We use the following libraries/projects:
* YOLOv5
* YOLOv8
* OpenCV

You can install all the requirements by the command:

~~~
pip install -r requirements.txt 
~~~

### 3 Normal usage

* <b>Create a library inside CancerDetector2 and put some images in it</b>

I.e. if you want to put your images into directory named "experiment":

~~~
mkdir experiment
cp {some_other directory} experiment
~~~

* <b>Download a pretrained model, or train one on your own (see below)</b>

Some links will be added here soon.

* <b>Create the initial masks using a neural network</b>

~~~
python generateMasksYolov5.py {some_yolov5_model.pt} {training_resolution} {image_files}
~~~
or
~~~~
python generateMasksYolov8.py {some_yolov8_model.pt} {image_files} 
~~~~

Say, you have downloaded our pretrained yolov5 large model, and copied 
some .jpg files into directory "experiment". Use:

~~~
python generateMasksYolov5.py yolov5l.cancer.pt 512 experiment/*.jpg
~~~

If you did it right, in directory "experiment" some ".mask.npy" and some ".vis.png"
files will be generated next to the original images. The former ones are the masks 
describing the blobs, the later ones are just for visualization. These are 
simple images, which show you what the neural network found.

* <b>Fix the masks, if needed; although the neural network's
output is typically close to perfect, it is usually needed to improve the
output a bit</b>

~~~
python maskEditorWindows.py {image_files}
~~~

i.e. if you want to improve the previous yolov5 results:

~~~
python maskEditorWindows.py experiment/*.jpg
~~~

You will see the previous results. You can add to the blobs by the 
left mouse button, delete by the right button and zoom in/out by
the mouse wheel. You get the next image by "n", the previous by "p",
jump 10 forward by "d" backward by "u". When you change the picture
the changes you made are saved. You can also save the changes by "s".
You can change the area you modify by "+" and "-" (don't forget to 
move the mouse a bit to see the change, this is a known issue).

Note that this was optimized for Windows, but works acceptable on 
Linux machines too. Linux-optimized version will be added.

* <b>Process the results</b>

~~~
python processResults.py experiment/*.jpg
~~~

The processing shows the number of blobs, the average size of blobs
in (micrometer**2) and the total size of the blobs. Note that in our
case, 200 micrometer was 110 pixels, so if that is not true anymore, you
need to update DEFAULT_PIXEL_IN_UM in the script. You may also need to modify 
variable DEFAULT_PIC_SIZE_IN_PIXEL, since it describes the normal 
resolution of the pics. 

Moreover, too small blobs are excluded; the threshold is defined in
micrometer**2 in variable SIZE_THRESHOLD_IN_UM2.

The output is generated in a Python dictionary, then prints it.
Moreover, it draws the results using matplotlib.pyplot.

### 4 How to train

You just need to train a plain YOLO model. To do that, please 
consult with [YOLOv5](https://github.com/ultralytics/yolov5) or
[YOLOv8](https://github.com/ultralytics/ultralytics). To make
things a bit simpler, hereby we present some little example:

* <b>Convert the mask files to darknet format</b>

Suppose that we're using bash:

~~~
for i in exp1/*.npy; do python numpy2yolo.py $i $i.txt; done
~~~

Now, build up the dataset directory by using the original images and
the freshly generated .txt files (these are the labels). This can
be put anywhere on a disc, later we will refer on it’s path.

The normal train/validation directory structure should be something 
like this:

~~~
dataset_root_dir
|
----> images
|     |
|     ------>train
|     |
|     ------>val
----> labels
      |
      ------>train
      |
      ------>val
~~~

Directory images must contain the training and validation images, while
labes must contain the .txt files we just created with numpy2yolo.py.

* <b>YOLOv5</b>

First, download YOLOv5 e.g. by using git:

~~~
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
cp {CancerDetectorDirectory}/YOLOv5TrainStuffs/* .
~~~

Next, you must edit "cancer-seg.yaml" we just copied into yolov5
directory to respect the dataset directory we built previously.

Training can be started like this (from the yolov5 directory):

~~~
python segment/train.py --data cancer-seg.yaml --epochs 300 --weights yolov5l-seg.pt --batch-size 10 --imgsz 512 --hyp hyp.yaml
~~~

* <b>YOLOv8</b>

First, create a new directory, and copy CancerDetector2/YOLOv5TrainStuffs/cancer-seg.yaml
to it. Next, edit the file with respect the path of the previously created
dataset directory.

When ultralytics was installed with pip (it's in requirements.txt, so should
be already installed), there should be a new command "yolo" added.

Training can be started like this:

~~~
yolo segment train data=cancer-seg.yaml model=yolov8s-seg.pt imgsz=512 batch=24 epochs=300 degrees=15.0
~~~


