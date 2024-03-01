from transformers import pipeline
from PIL import Image
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 255)  # BGR for yellow
stroke = 2  # thickness for rectangle

detection = pipeline("object-detection", model="facebook/detr-resnet-50")
stream = cv2.VideoCapture(0)
while True:
    (grabbed, frame) = stream.read()
    image = Image.fromarray(frame)
    results = detection(image)
    for item in results:
        box = [i for i in item['box'].values()]
        cv2.rectangle(
            frame, (box[0], box[1]), (box[2], box[3]), color, stroke
        )
        cv2.putText(
            frame, f'({item["label"]})',
            (box[0], box[1] - 8),
            font, 1, color, stroke,
            cv2.LINE_AA
        )

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # press q to break out of the loop
        break

stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
