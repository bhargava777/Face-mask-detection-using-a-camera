# import the necessary packages
import numpy as np
import imutils
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import cv2
import os

def face_mask(camera, faceNet, maskNet):
	# initialize our list of persons, their corresponding locations,
	# and the list of predictions from our face mask network
	persons = []
	points = []
	predictions = []

	# grab the dimensions of the camera and then construct a blob
	# from it
	(h, w) = camera.shape[:2]
	blob = cv2.dnn.blobFromImage(camera, 1.0, (224, 224),(104.0, 177.0, 123.0))


	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	

	#i=0
	
	# loop over the detections
	for i in range(0, detections.shape[2]):
	#
		# extract the confidence (i.e., probability) associated with
		# the detection
	#while(i<detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.4:
			# compute the (x, y)-coordinates of the bounding camera_frame for
			# the object
			camera_frame = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(first_X, first_Y, last_X, last_Y) = camera_frame.astype("int")

			# ensure the bounding camera_framees fall within the dimensions of
			# the camera
			(first_X, first_Y) = (max(0, first_X), max(0, first_Y))
			(last_X, last_Y) = (min(w - 1, last_X), min(h - 1, last_Y))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = camera[first_Y:last_Y, first_X:last_X]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding camera_framees to their respective
			# lists
			persons.append(face)
			points.append((first_X, first_Y, last_X, last_Y))
			#i=i+1

	# only make a predictions if at least one face was detected
	if len(persons) > 0:
		# for faster inference we'll make batch predictions on *all*
		# persons at the same time rather than one-by-one predictions
		# in the above `for` loop
		persons = np.array(persons, dtype="float32")
		predictions = maskNet.predict(persons, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (points, predictions)

# load our serialized face detector model from disk
prototxtPath = r"running.prototxt"
weightsPath = r"resultant_model.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("face_mask.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the cameras from the video stream
while True:
	# grab the camera from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	camera = vs.read()
	camera = imutils.resize(camera, width=400)

	# detect persons in the camera and determine if they are wearing a
	# face mask or not
	(points, predictions) = face_mask(camera, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (camera_frame, pred) in zip(points, predictions):
		# unpack the bounding camera_frame and predictions
		(first_X, first_Y, last_X, last_Y) = camera_frame
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding camera_frame and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding camera_frame rectangle on the output
		# camera
		cv2.putText(camera, label, (first_X, first_Y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(camera, (first_X, first_Y), (last_X, last_Y), color, 2)

	# show the output camera
	cv2.imshow("camera", camera)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
