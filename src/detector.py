import cv2
import imutils
import numpy as np
import tensorflow
from skimage.transform import pyramid_gaussian
from src.utils import sliding_window


class Detector:
    def __init__(self):
        self.svm_model_instance = None
        self.config = {
            'width': 300,
            'height': 300,
            'min_wdw_sz': (124, 124),
            'step_size': (10, 10),
            'downscale': 1.20
        }

        self.detect_fn = tensorflow.keras.models.load_model('../models/maskdetect-v1.12.h5')

    def preprocess_image(self, frame):
        im = imutils.resize(frame,
                            width=min(self.config['width'], frame.shape[1]),
                            height=min(self.config['height'], frame.shape[0]))
        im = np.expand_dims(im, axis=0)
        return im

    def predict_face_mask(self, _frame):
        im = self.preprocess_image(_frame)
        local_detections = []
        predict_results = self.detect_fn(im)

        print(predict_results)

        (face_mask, x_tl, y_tl, w, h) = predict_results[1]
        local_detections.append((x_tl, y_tl, w, h, face_mask))

        return local_detections


# Example usage
if __name__ == "__main__":
    image = cv2.imread('../img.png')
    detector = Detector()
    detections = detector.predict_face_mask(image)
    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(image, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)

    cv2.imshow("Face Mask Detector", image)
    cv2.waitKey(0)
