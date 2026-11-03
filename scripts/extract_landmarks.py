#extract_landmarks.py
#reads MP4 and saves 3D landmarks coordinates to a .npy file

import cv2
import mediapipe as mp
import numpy as np
import sys

def extract_landmarks(video_path, output_path):
	mp_pose = mp.solutions.pose
	pose = mp_pose.Pose(
		static_image_mode = False,
		model_complexity = 1,
		enable_segmentation = False,
		min_detection_confidence = 0.5
	)

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print("ERROR: Could not open Video:", video_path)
		return False

	all_landmarks = []

	frame_count = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		#convert BGR (openCV) to RGB (mediapipe)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = pose.process(rgb)

		if results.pose_world_landmarks:
			#extract 33 landmarks, with x,y,z
			frame_data = []
			for lm in results.pose_world_landmarks.landmark:
				frame_data.append([lm.x, lm.y, lm.z])
			all_landmarks.append(frame_data)
		else:
			#if no pose detected, repeat last knowon Pose
			if all_landmarks:
				all_landmarks.append(all_landmarks[-1])
			else:
				#no pose at all - fill with zeroes
				all_landmarks.append([[0.0, 0.0, 0.0]] * 33)

		frame_count += 1
		if frame_count % 30 == 0:
			print(f"Processed {frame_count} frames...")

	cap.release()
	pose.close()

	#save as numpy arrays: shape (num_frames, 33, 3)
	arr = np.array(all_landmarks, dtype=np.float32)
	np.save(output_path, arr)
	print(f"Saved landmarks to {output_path}, shape: {arr.shape}")
	return True

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: python extract_landmarks.py input.mp4 > output.npy")
		sys.exit(1)
	extract_landmarks(sys.argv[1], sys.argv[2])