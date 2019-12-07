'''
    This program loads a Tensorflow object detection model and detects
    counter_terrorist / terrorist / ambiguous objects in the specified screen
    region. It paints bounding boxes and class labels around the identified objects (humanoids).

    The queue-based parallelization idea is credited to Evan Juras, who in turn
    was inspired by Dat Tran
    https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py
'''

# Import packages
import multiprocessing, os
import cv2
import numpy as np
from mss import mss

import keras_retinanet
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.visualization import draw_box, draw_caption

# use this to change which GPU to use
gpu = 0; setup_gpu(gpu)

# Custom helper scripts
from src.utils import load_trained_model, load_label_map, intersection_over_union


def GRABMSS_screen(q, monitor):
    with mss() as sct:
        while True:
            # Get raw pixels from the screen, save as Numpy array *without* the alpha channel
            img = np.array(sct.grab(monitor))[:,:,:-1]
            # To get real color we do this:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Add the 'batch' dimension
            img = np.expand_dims(img, axis=0)
            q.put_nowait(img)
            q.join()


def make_predictions(model_path, q, detection_threshold=0.5, overlap_threshold=0.4):
    '''
        Grab image from queue, make predictions and overlay bounding boxes onto it.
        Assumes that image is preprocessed into a form ingestible by predict_on_batch()

        Args:
            model:
            queue::Queue  Contains images as numpy.array (batch_size, height, width); batch_size is usually 1
            detection_threshold::float
            overlap_threshold::float

        Modifies 'image' in-place, draws bounding boxes (if any) on top of
        existing array.
    '''

    # Load model into memory.
    model = load_trained_model(model_path, backbone_name='resnet50')
    labels_to_names = load_label_map()

    while True:
        if not q.empty():
            image = q.get_nowait()
            q.task_done()

        # process image
        boxes, scores, labels = model.predict_on_batch(image)

        # Filter out non-detections, using the classification probability
        # Note that we want to keep the last dimension of boxes, which contains the bounding box cooordinates
        score_mask = (scores > detection_threshold)
        scores = scores[score_mask]
        labels = labels[score_mask]
        boxes = boxes[0, score_mask[0], :]

        ## Apply non-max suppression
        for iA,(boxA,scoreA) in enumerate(zip(boxes, scores)):
            if scoreA > 0.0:
                try:
                    # Search forward for overlapping boxes
                    suppression_candidates = [(iA, scoreA)]
                    for iB,(boxB,scoreB) in enumerate(zip(boxes[iA+1:, :], scores[iA+1:]), start=iA+1):
                        if intersection_over_union(boxA, boxB) > overlap_threshold:
                            suppression_candidates.append((iB, scoreB))

                    # Non-max suppression by setting score to 0.0
                    if len(suppression_candidates) > 1:
                        suppression_candidates.sort(key=lambda x: x[-1])
                        for target, _ in suppression_candidates[:-1]:
                            scores[target] = 0.0

                except IndexError:
                    # catch iA+1 when it goes out of bound
                    # Allegedly try-except blocks are ubiquitous for control-flow in Python
                    pass

        ## Make use of bounding boxes and predicted classes
        ## e.g. Draw bounding boxes
        p_screen = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)

        for box, score, label in zip(boxes, scores, labels):
            if (score <= detection_threshold): continue
            color = label_color(label)
            b = box.astype(int)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_box(p_screen, b, color=color, thickness=2)
            draw_caption(p_screen, b, caption)

        cv2.imshow('Object detector', p_screen)

        #cv2.imshow('window',cv2.cvtColor(np.array(p_screen),cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__=="__main__":
    # Define screen-grab parameters
    mon = {'top': 40, 'left': 10, 'width': 700, 'height': 500}
    queue = multiprocessing.JoinableQueue(maxsize=50)

    # Give path to trained Keras model
    model_path = os.path.join('snapshots', 'resnet50', 'resnet50_csv_30.h5')

    # creating new processes
    p1 = multiprocessing.Process(target=GRABMSS_screen, args=(queue, mon))
    p2 = multiprocessing.Process(target=make_predictions, args=(model_path, queue, 0.5, 0.4))

    # starting our processes
    p1.start()
    p2.start()

    p1.join()
    p2.join()
