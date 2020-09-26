from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import detect_mask


class VideoProcessor:
    def __init__(self, args):
        self.confidence = args["confidence"]
        print("[INFO] loading face detector model...")
        proto_txt_path = os.path.sep.join([args["face"], "deploy.prototxt"])
        weights_path = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
        self.face_net_model = cv2.dnn.readNet(proto_txt_path, weights_path)
        self.detect_mask_feature = detect_mask.DetectMaskProcessor(args)

        print("[INFO] starting video stream...")
        self.video_stream_instance = VideoStream(src=0).start()

    def detect_face_in_frame(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        self.face_net_model.setInput(blob)
        detections = self.face_net_model.forward()

        faces = []
        entity_locations = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                entity_locations.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")

        return entity_locations, faces

    def start_video_composer(self):
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = self.video_stream_instance.read()
            frame = imutils.resize(frame, width=400)

            mask_predictions_results = []
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (entity_locations, faces) = self.detect_face_in_frame(frame)

            # detect mask on each face in the array
            if len(faces) > 0:
                mask_predictions_results = self.detect_mask_feature.detect_and_predict_mask(faces)

            # loop over the detected face locations and their corresponding
            # locations
            for (index, face_box) in enumerate(entity_locations):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = face_box
                (mask, withoutMask) = mask_predictions_results[index]

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.video_stream_instance.stop()
