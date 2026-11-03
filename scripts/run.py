#run,py
#master script: mp4 > landmarks > rotations > fbx
#usage: python run.py input.mp4 output.fbx

import subprocess
import sys
import os

#path setup, diff when inside pyinstaller 
if getattr(sys, 'frozen', False):
	BUNDLE_DIR = sys._MEIPASS                          # temp extraction folder
	EXE_DIR = os.path.dirname(sys.executable)          # where the exe lives
else:
	BUNDLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	EXE_DIR = BUNDLE_DIR

SCRIPTS = os.path.join(BUNDLE_DIR, "scripts")
ASSETS  = os.path.join(BUNDLE_DIR, "assets")

def find_blender():
	candidates = [
		os.path.join(EXE_DIR, "BlenderPortable", "blender.exe"),
		os.path.join(EXE_DIR, "assets", "BlenderPortable", "blender.exe"),
		os.path.join(BUNDLE_DIR, "assets", "BlenderPortable", "blender.exe"),
		os.path.join(EXE_DIR, "dist", "BlenderPortable", "blender.exe"),
	]
	for c in candidates:
		if os.path.exists(c):
			return c
	raise FileNotFoundError(
		"blender.exe not found. Looked in:\n  " + "\n  ".join(candidates))

#import extraction function to call it directly	
sys.path.insert(0, SCRIPTS)
from extract_landmarks import extract_landmarks as do_extract

def main(video_path, output_fbx):
	output_fbx = os.path.abspath(output_fbx)
	os.makedirs(os.path.dirname(output_fbx), exist_ok=True)

	#landmarks
	print("=" * 60)
	print("STEP: Extract Landmarks")
	landmarks_path = os.path.join(os.path.dirname(output_fbx), "landmarks.npy")
	if not do_extract(video_path, landmarks_path):
		raise RuntimeError("Landmark extraction failed")

	#blender export
	print("=" * 60)
	print("STEP: Export FBX")
	blender = find_blender()
	export_script = os.path.join(SCRIPTS, "export_fbx.py")
	rig_path = os.path.join(ASSETS, "template_rig.fbx")

	cmd = [blender, "--background", "--python", export_script, "--",
	       rig_path, landmarks_path, output_fbx]
	print("CMD:", " ".join(cmd))

	result = subprocess.run(cmd, capture_output=True, text=True)
	print(result.stdout)
	if result.stderr:
		print(result.stderr)

	failed = (result.returncode != 0
	          or "Traceback" in (result.stderr or "")
	          or "RuntimeError" in (result.stderr or ""))
	if failed:
		raise RuntimeError("Blender export failed")

	if not os.path.exists(output_fbx):
		raise RuntimeError("Blender finished but did not produce " + output_fbx)

	print("\nComplete! Output:", output_fbx)
	return output_fbx

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("Usage: python run.py input.mp4 output.fbx")
		sys.exit(1)
	main(sys.argv[1], sys.argv[2])