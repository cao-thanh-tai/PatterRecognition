# build model hand detection with yolov8

from ultralytics import YOLO


class HandDetectionModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def train(self, **kwargs):
        """Train the YOLO model."""
        return self.model.train(**kwargs)

    def validate(self, **kwargs):
        """Validate the YOLO model."""
        return self.model.val(**kwargs)

    def predict(self, image, **kwargs):
        """
        Predict hand bounding boxes in the given image.

        Args:
            image (PIL.Image or np.ndarray): Input image.

        Returns:
            list: List of predicted bounding boxes.
        """
        return self.model.predict(image, **kwargs)

    def export(self, **kwargs):
        """Export the trained model."""
        return self.model.export(**kwargs)