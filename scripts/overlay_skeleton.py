#overlay_skeleton.py
#draws mediapipe landmarks on top of video to verify accuracy

import cv2
import numpy as np
import os
import sys

#mediapipe connections (landmarks joined by line)
CONNECTIONS = [
	(11,12), (11,13), (13,15), (12,14), (14, 16),
	(11,23), (12,24), (23,24),
	(23,25), (25,27), (24,26), (26,28),
	(27,29), (29,31), (28,30), (30,32),
	(27,31), (28,32),
	(0,11), (0,12),
	(9,0),
]

def derive_paths(video_path):
	root, _ = os.path.splitext(video_path)
	return root + ".npy", root + "_overlay.mp4"

def make_overlay(video_path, npy_path, output_path):
	data = np.load(npy_path)
	cap = cv2.VideoCapture(video_path)
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

	frame_idx = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if frame_idx >= data.shape[0]:
			break

		lms = data[frame_idx]

		#mediapipe world landmarks are meters around the hip
		#map x > screenx, y > screeny by scaling and centering
		#assume person is center
		scale = min(w,h) * 0.7
		cx,cy = w/2, h/2

		def project(lm):
			return int(cx + lm[0] * scale), int(cy + lm[1] * scale)

		#draw lines
		for a, b in CONNECTIONS:
			pa = project(lms[a])
			pb = project(lms[b])
			cv2.line(frame, pa, pb, (0,255,0),2)

		#draw dots
		for i in range(33):
			p = project(lms[i])
			cv2.circle(frame, p, 4, (0 ,0 ,255), -1)

		cv2.putText(frame, f"Frame [{frame_idx}]", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

		writer.write(frame)
		frame_idx += 1

	cap.release()
	writer.release()
	print(f"Saved overlay video to {output_path} -- {frame_idx} frames")

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python overlay_skeleton.py test.mp4 landmarks.npy overlay.mp4")
		sys.exit(1)
	video = sys.argv[1]
	npy, out = derive_paths(video)
	if not os.path.exists(npy):
		print(f"ERROR: {npy} not found. Run extract_landmarks.py on video FIRST!")
		sys.exit(1)
	make_overlay(video, npy, out)
