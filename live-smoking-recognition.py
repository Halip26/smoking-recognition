import cv2
import imutils
import numpy as np
import argparse
import pickle
import time
import os

from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def main(args):
    # Load face detector model
    print("[INFO] Loading face detector...")
    protoPath = os.path.join("weights", "deploy.prototxt")
    modelPath = os.path.join("weights", "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load smoking recognition model and label encoder
    print("[INFO] Loading smoking recognition...")
    model = load_model("weights/smoking.keras")
    le = pickle.loads(open("weights/le.pickle", "rb").read())

    # Start video stream
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # Loop over frames from video stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=1024)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
 
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > args["face_confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_gray = np.zeros_like(frame)
                img_gray[:, :, 0] = gray
                img_gray[:, :, 1] = gray
                img_gray[:, :, 2] = gray
                face = img_gray[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                predict = model.predict(face)[0]
                j = np.argmax(predict)
                label = le.classes_[j]

                label = "{}: {:.4f}%".format(label, predict[j] * 100)
                cv2.putText(
                    frame,
                    label,
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        cv2.imshow("Smoking_Recognition_v1.0", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # generate the image
        output_directory = "images/output/"
        os.makedirs(output_directory, exist_ok=True)

        cv2.imwrite("images/output/live-recognition-result.jpg", frame)

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c",
        "--face_confidence",
        type=float,
        default=0.7,
        help="minimum probability to filter weak detections",
    )

    args = vars(ap.parse_args())
    main(args)
