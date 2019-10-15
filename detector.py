import numpy as np
import argparse
import cv2
from imutils.video import VideoStream
import time
import imutils

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
PROTOTEXT = 'MobileNetSSD_deploy.prototxt'
CAFFEMODEL = 'MobileNetSSD_deploy.caffemodel'

# load our serialized model from disk
# print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTEXT, CAFFEMODEL)

# print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=640)
	# frame_copy = frame.copy()


	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
		(300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	# print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
	
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the `detections`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	
			# display the prediction
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			# print(CLASSES[idx])
			if CLASSES[idx] == "person":
				# print("[INFO] {}".format(label))
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	
cv2.destroyAllWindows()
vs.stop()