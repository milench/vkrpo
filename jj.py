import datetime
import cv2
import supervision as sv
from ultralytics import YOLO
#from helper import create_video_writer

THRESHOLD = 0.5

video = cv2.VideoCapture('vid.mp4')

#writer = create_video_writer(video, "output.mp4")
data = [[835, 15, 1054, 612, 0.94, 0], [549, 260, 679, 623, 0.91, 0], [308, 370, 589, 629, 0.84, 13]]
model = YOLO('runs/detect/boat_detect/weights/best.pt')
GREEN = (0, 255, 0)
while True:
    start = datetime.datetime.now()

    ret, frame = video.read()

    if not ret:
        break
    detections = model(frame)[0]
 # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]

        # filter out weak detections by ensuring the
        # confidence is greater than the minimum confidence
        if float(confidence) < THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
 # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    #writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video.release()
#writer.release()
cv2.destroyAllWindows()
