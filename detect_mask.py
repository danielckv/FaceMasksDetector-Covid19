from tensorflow.keras.models import load_model


class DetectMaskProcessor:

    def __init__(self, args):
        print("[INFO] loading face mask detector model...")
        mask_net = load_model(args["model"])
        self.maskNet = mask_net

    def detect_and_predict_mask(self, faces):
        mask_predictions = self.maskNet.predict(faces, batch_size=32)
        return mask_predictions
