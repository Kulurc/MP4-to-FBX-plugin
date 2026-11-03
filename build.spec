#build.spec

import sys
import os
from PyInstaller.utils.hooks import collect_all

mediapipe_datas = collect_all('mediapipe')[0]

a = Analysis(
	['scripts/app.py'],
	pathex = [],
	binaries = [],
	datas = [
		('scripts/extract_landmarks.py', 'scripts'),
		('scripts/export_fbx.py', 'scripts'),
		('scripts/run.py', 'scripts'),
		('assets/template_rig.fbx', 'assets'),
	] + mediapipe_datas, 
	hiddenimports = ['mediapipe', 'cv2', 'numpy'],
	hookspath = [],
	runtime_books = [],
	excludes = [],
	noarchive = False,
)

pyz = PYZ(a.pure)

exe = EXE(
	pyz,
	a.scripts,
	a.binaries,
	a.datas,
	[],
	name = 'MP4 to FBX',
	debug = False,
	bootloader_ignore_signals = False,
	strip = False,
	upx = True,
	console = True,
	icon = None,
)