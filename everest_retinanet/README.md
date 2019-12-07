# Identify Friend-or-Foe via Object Detection
Training and running a RetinaNet object detection model that distinguishes
between `counter_terrorist`, `terrorist`, and `ambiguous` classes in the games
*Counter Strike: 1.6* and *Counter Strike: Global Offensive*.  

Retinanet implementation courtesy of [Fizyr](https://github.com/fizyr) through
Apache License 2.0. Everest Law ("I") wrote all the scripts provided in this directory.


## Set Up
To train a model, install the [retinanet](https://github.com/fizyr/keras-retinanet) package and
all its [prerequisites](https://github.com/fizyr/keras-retinanet/blob/master/requirements.txt). I wrote the code using Tensorflow 1.14/1.15 + Python
3.7.5 on Windows 10, but it shouldn't be OS-specific.

To view object detection results on-screen, install the [mss](https://python-mss.readthedocs.io/installation.html)
and [OpenCV2](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html) libraries.


## Preparing Data
My team has manually labeled bounding boxes in ~3000 images using [LabelImg](https://github.com/tzutalin/labelImg),
and then saved our annotations in the PascalVOC XML format. In order for Retinanet to
consume the images, we must collate all annotations into a single
CSV file; see [here](https://github.com/fizyr/keras-retinanet#csv-datasets) for details.

Run `retinanet_xml_to_csv.py` to generate the CSV; remember to supply the directory
paths containing images and XMLs. Include the `--train_test_split` argument to
produce a 80-20 split based on unique images (**not** bounding boxes). For instance:
```
python retinanet_xml_to_csv.py data/imgs/Everest data/labels/Everest --train_test_split
```

## Initiate Training
The easiest way is through the `retinanet-train`
command-line utility, which should sit in your PATH after installing the `retinanet` package.
Examples are provided in `sample_commands.txt` --- run them in a console!

Notice how the examples specify a Resnet50 "backbone". The authors of `retinanet`
only provided [COCO pretrained weights](https://github.com/fizyr/keras-retinanet/releases/tag/0.5.1)
based on Resnets, at depths 50, 101, and 152. I took the shallowest network, which
supposedly runs the quickest during inference.

In trying to minimize run-time overhead, I have also trained a MobileNet128 model
from scratch --- i.e. with ImageNet instead of COCO weights. Unfortunately, the training
loss refused to decrease pass a certain point, and the inference precision is terribly low.
Maybe Mobilenet would work *after* pre-training on COCO --- but for now that remains a mystery.

## View Results of Training
### Live inference
The script `liveshow.py` tracks objects in a given region of the screen, and
paints bounding boxes over them wherever applicable. Run as `python liveshow.py`
--- try it while playing *Counter Strike* in window mode!

### Mean Average Precision (mAP)
The `retinanet-evaluate` command-line utility consumes the CSV files generated
and outputs class-specific + overall mAP values. See `sample_commands.txt`. Remember
to specify intersection-over-union and score thresholds!
