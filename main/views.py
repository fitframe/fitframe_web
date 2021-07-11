from django.shortcuts import render
from django.http.response import StreamingHttpResponse
import cv2
import numpy as np
import mediapipe as mp
import threading

from django.utils.baseconv import base64
from django.views.decorators import gzip


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)

	def __del__(self):
		self.video.release()

	def get_frame(self):
		success, image = self.video.read()
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		frame_flip = cv2.flip(image,1)
		ret, jpeg = cv2.imencode('.jpg', pose_gen(image))
		return jpeg.tobytes()


def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')

def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')

def show(request):
	return render(request, 'main/video.html')


def pose_gen(frame):
	counter = 0
	stage = None

	pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

	# Recolor image to RGB
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False

	# Make detection
	results = pose.process(image)

	# Recolor back to BGR
	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	# Extract landmarks
	try:
		landmarks = results.pose_landmarks.landmark

		# Get coordinates
		shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
					  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
		elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
				   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
		rist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
				  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
		shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
					  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
		elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
				   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
		wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
				   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

		# Calculate angle
		angle_main_L = calculate_angle(shoulder_R, shoulder_L, elbow_L)
		angle_main_R = calculate_angle(shoulder_L, shoulder_R, elbow_R)

		# Visualize angle
		cv2.putText(image, str(angle_main_L),
					tuple(np.multiply(elbow_L, [640, 480]).astype(int)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
					)
		cv2.putText(image, str(angle_main_R),
					tuple(np.multiply(elbow_R, [640, 480]).astype(int)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
					)

		# Curl counter logic
		if angle_main_L > 109 and angle_main_R > 109:  # 160
			stage = "down"
		if angle_main_L < 111 and angle_main_R < 111 and stage == 'down':  # 30
			stage = "up"
			counter += 1
			print(counter)

	except:
		pass

	# Render curl counter
	# Setup status box
	cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

	# Rep data
	cv2.putText(image, 'REPS', (15, 12),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.putText(image, str(counter),
				(10, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

	# Stage data
	cv2.putText(image, 'STAGE', (65, 12),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.putText(image, stage,
				(60, 60),
				cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

	# Render detections
	mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
							  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
							  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
							  )

	return image