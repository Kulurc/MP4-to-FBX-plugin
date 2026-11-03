#test_video.py
#runs full pipeline (extract > overlay) on any video
#usage: python test_video.py path\to\video.mp4

import subprocess
import sys
import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))

def run(label, cmd):
	print(f"\n=== {label} ===")
	print(">", " ".join(cmd))
	result = subprocess.run(cmd)
	if result.returncode != 0:
		print(f"FAILED: {label}")
		sys.exit(1)

def main(video):
	video = os.path.abspath(video)
	if not os.path.exists(video):
		print(f"Video not found: {video}")
		sys.exit(1)

	root, _ = os.path.splitext(video)
	npy = root + ".npy"
	overlay = root + "_overlay.mp4"

	#1. extract landmarks
	if os.path.exists(npy):
		print(f"Landmark already exists: {npy} --Skipping extraction...")
	else:
		run("Extract Landmarks",
		[sys.executable, os.path.join(SCRIPTS, "extract_landmarks.py"),
		video, npy])

	#2: overlay
	run("Make overlay",
	[sys.executable, os.path.join(SCRIPTS, "overlay_skeleton.py"),
	video])

	print("\nDone. Open: ", overlay)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(r"Usage: python test_video.py path\to\video.mp4")
		sys.exit(1)
	main(sys.argv[1])