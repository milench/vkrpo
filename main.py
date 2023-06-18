import cv2
from ultralytics import YOLO
import supervision as sv
import datetime

THESHHOLD = 0.2

boat_class_index = 8

class BoatTracking:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = self.yolo_model()

        self.box = sv.BoxAnnotator(
            color=sv.Color(r=128, g=0, b=0),
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

    def fps_checker(self, p_time, c_time):
        fps = (c_time - p_time).total_seconds()
        fps = f"FPS: {1 / fps:.2f}"
        return fps


    def yolo_model(self):
        model = YOLO('yolov8n.pt')
        model.classes = [8]
        model.predict(source="capture_index", show=False, stream=True, classes=8)
        model.fuse()
        return model

    def __call__(self):
        video = cv2.VideoCapture(self.capture_index)
        assert video.isOpened()

        video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        while True:
            p_time = datetime.datetime.now()  # start fps counter
            success, frame = video.read()
            assert success
            results = self.model(frame)[0]
            detections = sv.Detections.from_yolov8(results)
            labels = [
                f"{confidence:0.2f}"
                for confidence
                in detections.confidence
            ]
            frame = self.box.annotate(scene=frame, detections=detections, labels=labels)

            c_time = datetime.datetime.now()  # end fps counter
            fps = self.fps_checker(p_time, c_time)  # fps finder

            cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
            cv2.imshow('result', frame)
            for result in results:

                # classification
                result.probs  # cls prob, (num_class, )

            if cv2.waitKey(1) == ord("q"):
                break
        video.release()
        cv2.destroyAllWindows

tracking = BoatTracking(capture_index='vid5.mp4')
tracking()
