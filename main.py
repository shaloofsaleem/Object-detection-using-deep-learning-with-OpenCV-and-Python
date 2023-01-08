import cv2

# capturing video from default camera
cap = cv2.VideoCapture(0)

# converting the names from coco.names file to list using rsplit()
className = []
classFile = 'coco.names'
with open(classFile, 'rt') as file:
    className = file.read().rsplit('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 125.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    _, img = cap.read()
    # confs :- Accuracy ratio
    classIds, confs, bbox = net.detect(img, confThreshold=0.6)

    print(classIds, confs, bbox)
    # if classIds != 0 means, if no object detected then do nothing
    if len(classIds) != 0:
        for classId, confs, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, (0, 255, 0), 2)
            # Here our className value starts from 0, so we need to subtract 1 for getting correct result
            cv2.putText(img, className[classId - 1], (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

