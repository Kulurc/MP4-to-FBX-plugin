#export_fbx.py
#runs inside blender, reads mediapipe landmarks, smooths and compute bone rotations relative to rig
#applies keyframes and exports FBX

import bpy
import numpy as np
import sys
import os
from mathutils import Vector, Quaternion

#mediapipe landmark indices
LM = {
	'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13,   'right_elbow': 14,
    'left_wrist': 15,   'right_wrist': 16,
    'left_hip': 23,     'right_hip': 24,
    'left_knee': 25,    'right_knee': 26,
    'left_ankle': 27,   'right_ankle': 28,
}

#names based off blender's human meta-rig
BONE_MAP = {
    'upper_arm.L': ('left_shoulder', 'left_elbow'),
    'upper_arm.R': ('right_shoulder', 'right_elbow'),
    'forearm.L':   ('left_elbow', 'left_wrist'),
    'forearm.R':   ('right_elbow', 'right_wrist'),
    'thigh.L':     ('left_hip', 'left_knee'),
    'thigh.R':     ('right_hip', 'right_knee'),
    'shin.L':      ('left_knee', 'left_ankle'),
    'shin.R':      ('right_knee', 'right_ankle'),
}

#one euro filter
class OneEuroFilter:
	def __init__(self, min_cutoff = 1.0, beta = 0.05, d_cutoff = 1.0):
		self.min_cutoff = min_cutoff
		self.beta = beta
		self.d_cutoff = d_cutoff
		self.x_prev = None
		self.dx_prev = None

	def __call__(self, x, dt):
		if self.x_prev is None:
			self.x_prev = x
			self.dx_prev = np.zeros_like(x)
			return x

		#estimate derivative
		dx = (x - self.x_prev) / dt
		
		#smooth derivative
		alpha_d = self._alpha(self.d_cutoff, dt)
		dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

		#compute cutoff freq
		cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
		alpha = self._alpha(cutoff, dt)

		#filter signal
		x_hat = alpha * x + (1 - alpha) * self.x_prev

		self.x_prev = x_hat
		self.dx_prev = dx_hat
		return x_hat
	
	def _alpha(self, cutoff, dt):
		tau = 1.0 / (2 * np.pi * cutoff)
		return 1.0 / (1.0 + tau / dt)


def smooth_1d(signal, fps = 30, min_cutoff = 1.0, beta = 0.05):
	#smooth a 1D numpy array with the one euro filter
	dt = 1.0 / fps
	f = OneEuroFilter(min_cutoff = min_cutoff, beta = beta)
	out = np.zeros_like(signal)
	for i, v in enumerate(signal):
		out[i] = f(v, dt)
	return out


def smooth_landmarks(landmarks, fps = 30):
	"""Apply One Euro Filter to every landmark coordinate."""
	num_frames, num_landmarks, _ = landmarks.shape
	dt = 1.0 / fps

	filters = {}
	smoothed = np.zeros_like(landmarks)

	for lm_idx in range(num_landmarks):
		for axis in range(3):
			key = (lm_idx, axis)
			filters[key] = OneEuroFilter(min_cutoff = 1.0, beta = 0.007)

	for f in range(num_frames):
		for lm_idx in range(num_landmarks):
			for axis in range(3):
				val = landmarks[f, lm_idx, axis]
				smoothed[f, lm_idx, axis] = filters[ (lm_idx, axis)](val,dt)

	return smoothed


#coordinate conversion: mediapipe > blender
#mp: x right, y down, z towards camera
#blender: x right, y in, z up
def mp_to_blender(v):
	return Vector((float(v[0]), float(v[2]), -float(v[1])))

#scene helpers
def clear_scene():
	bpy.ops.object.select_all(action = 'SELECT')
	bpy.ops.object.delete()

def load_rig(rig_path):
	if not os.path.exists(rig_path):
		raise RuntimeError(f"Template rig not found: {rig_path}")
	bpy.ops.import_scene.fbx(filepath = rig_path)
	for obj in bpy.context.selected_objects:
		if obj.type == 'ARMATURE':
			return obj
	raise RuntimeError("No armature found in template rig")

#compute per-frame bone-local rotations
def compute_bone_rotations(armature, landmarks, fps = 30):
	landmarks = smooth_landmarks(landmarks, fps)
	num_frames = landmarks.shape[0]
	bones = armature.data.bones

	#get rest-pose info for each mapped bone
	rest_info = {}
	for bone_name in BONE_MAP:
		bone = bones.get(bone_name)
		if bone is None:
			print(f">>>WARNING: bone missing from rig: {bone_name}")
			continue

		rest_dir = (bone.tail_local - bone.head_local).normalized()
		R_rest_world = bone.matrix_local.to_quaternion()

		if bone.parent:
			M_rest_rel = bone.parent.matrix_local.inverted() @ bone.matrix_local
			R_rest_rel = M_rest_rel.to_quaternion()
			parent_name = bone.parent.name
		else:
			R_rest_rel = None
			parent_name = None

		rest_info[bone_name] = {
			'rest_dir': rest_dir,
			'R_rest_world': R_rest_world,
			'R_rest_rel': R_rest_rel,
			'parent_name': parent_name,
		}

	#process parents before children
	def depth(name):
		b = bones.get(name); d = 0
		while b and b.parent:
			d += 1; b = b.parent
		return d

	ordered = sorted(rest_info.keys(), key = depth)
	print(f">>>Computing {len(ordered)} bones over {num_frames} frames...")

	result = {name: [] for name in rest_info}

	for frame_idx in range(num_frames):
		frame_lms = landmarks[frame_idx]
		world_rot = {}

		for bone_name in ordered:
			info = rest_info[bone_name]
			parent_lm, child_lm = BONE_MAP[bone_name]

			p_bl = mp_to_blender(frame_lms[LM[parent_lm]])
			c_bl = mp_to_blender(frame_lms[LM[child_lm]])
			delta = c_bl - p_bl

			if delta.length < 1e-6:
				result[bone_name].append(Quaternion((1,0,0,0)))
				world_rot[bone_name] = info['R_rest_world']
				continue

			d_world = delta.normalized()
			q_delta = info['rest_dir'].rotation_difference(d_world)
			R_bone_world = q_delta @ info['R_rest_world']

			if info['parent_name'] and info['parent_name'] in world_rot:
				R_parent = world_rot[info['parent_name']]
				R_basis = (info['R_rest_rel'].inverted() @ R_parent.inverted() @ R_bone_world)
			else:
				R_basis = info['R_rest_world'].inverted() @R_bone_world

			result[bone_name].append(R_basis)
			world_rot[bone_name] = R_bone_world

	return result

def compute_spine_rotation(armature, landmarks, fps = 30):
	"""Compute spine twist (Yaw) + forward lean (Pitch) + side bend (Roll) from the shoulder-line direction"""
	landmarks = smooth_landmarks(landmarks, fps)
	num_frames = landmarks.shape[0]
	bones = armature.data.bones

	#gather every spine bone in rig
	spine_names = sorted(
		[b.name for b in bones if b.name == 'spine' or b.name.startswith('spine.')],
	)
	if not spine_names:
		print("[!] WARNING: no spine bones found in rig")
		return {}

	n = len(spine_names)
	print(f">>> Spine Chain: {spine_names} ({n} bones)")

	#compute shoulder-line yaw per frame in blender's coordinate frame
	yaws = np.zeros(num_frames, dtype = np.float64)
	pitches = np.zeros(num_frames, dtype = np.float64)
	rolls = np.zeros(num_frames, dtype = np.float64)

	for f in range(num_frames):
		l_sh = mp_to_blender(landmarks[f, 11]) #left shoulder
		r_sh = mp_to_blender(landmarks[f, 12]) #right shoulder

		l_hip = mp_to_blender(landmarks[f, 23])
		r_hip = mp_to_blender(landmarks[f, 24])

		mid_sh = (l_sh + r_sh) / 2
		mid_hip = (l_hip + r_hip) / 2
		up = mid_sh - mid_hip

		if up.length < 1e-16:
			continue
		up = up.normalized()

		#yaw from shoulder line
		shoulder_vec = r_sh - l_sh
		yaw = np.arctan2(shoulder_vec.y, shoulder_vec.x)
		yaws[f] = yaw

		#de-yaw the up vector so pitch and roll are measured in body aligned frame
		cy, sy = np.cos(-yaw), np.sin(-yaw)
		ux = cy * up.x - sy * up.y
		uy = sy * up.x + cy * up.y
		uz = up.z

		#pitch - forward lean around body's left-right axis
		pitches[f] = -np.arctan2(-uy, uz)

		#roll - side lean around body's front-back axis
		rolls[f] = np.arctan2(ux, uz)

	#ref frames = first frame
	yaws -= yaws[0]
	pitches -= pitches[0]
	rolls -= rolls[0]

	#unwrap to prevent jumps, then smooth_1d
	yaws = np.unwrap(yaws)
	pitches = np.unwrap(pitches)
	rolls = np.unwrap(rolls)

	yaws = smooth_1d(yaws, fps = fps, min_cutoff = 1.0, beta = 0.01)
	pitches = smooth_1d(pitches, fps = fps, min_cutoff = 1.0, beta = 0.01)
	rolls = smooth_1d(rolls, fps = fps, min_cutoff = 1.0, beta = 0.01)

	result = {name: [] for name in spine_names}
	for f in range(num_frames):
		#rotate around the bone's LOCAL y axis
		q_yaw = Quaternion((0.0, 1.0, 0.0), float(yaws[f] / n))
		q_pitch = Quaternion((1.0, 0.0, 0.0), float(pitches[f] / n))
		q_roll = Quaternion((0.0, 0.0, 1.0), float(rolls[f] / n))

		#blend first, twist last
		q_total = q_yaw @ q_pitch @q_roll

		for name in spine_names:
			result[name].append(q_total.copy())

	return result



#translate whole rig vertically so feet stay grounded
def apply_root_translation(armature, landmarks, fps=30):
    """Translate the rig vertically so feet stay grounded."""
    num_frames = landmarks.shape[0]

    lowest_z = []
    for f in range(num_frames):
        #check ankles, heels and toes to find lowest point
        points = [
            mp_to_blender(landmarks[f, 27]), mp_to_blender(landmarks[f, 28]), # ankles
            mp_to_blender(landmarks[f, 29]), mp_to_blender(landmarks[f, 30]), # heels
            mp_to_blender(landmarks[f, 31]), mp_to_blender(landmarks[f, 32])  # toes
        ]
        #lowest point is the most negative Z
        lowest_z.append(min(p.z for p in points))

    ground = min(lowest_z)
    
    # Smooth the offsets to prevent jitter
    z_arr = np.array(lowest_z, dtype=np.float64)
    z_smooth = smooth_1d(z_arr, fps=fps, min_cutoff=1.0, beta=0.01)
    
    offsets = [ground - z for z in z_smooth]
    
    print(f">>> Root translation range: {min(offsets):.3f} to {max(offsets):.3f} m")

    base_loc = armature.location.copy()

    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    for f in range(num_frames):
        armature.location = (
            base_loc.x,
            base_loc.y,
            base_loc.z + offsets[f],
        )
        armature.keyframe_insert(data_path="location", frame=f + 1)

    armature.location = base_loc



#apply keyframes
def apply_rotations(armature, rotations, fps = 30):
	scene = bpy.context.scene
	scene.render.fps = fps

	bpy.ops.object.select_all(action = 'DESELECT')
	armature.select_set(True)
	bpy.context.view_layer.objects.active = armature
	bpy.ops.object.mode_set(mode = 'POSE')

	max_frames = max((len(v) for v in rotations.values()), default = 0)

	for frame_idx in range(max_frames):
		scene.frame_set(frame_idx + 1)
		for bone_name, quats in rotations.items():
			if frame_idx >= len(quats):
				continue
			pb = armature.pose.bones.get(bone_name)
			if pb is None:
				continue
			pb.rotation_mode = 'QUATERNION'
			pb.rotation_quaternion = quats[frame_idx]
			pb.keyframe_insert(data_path = "rotation_quaternion", frame = frame_idx + 1)

	bpy.ops.object.mode_set(mode = 'OBJECT')

def export_fbx(output_path):
	out_dir = os.path.dirname(output_path)

	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok = True)

	bpy.ops.export_scene.fbx(
		filepath = output_path,
		use_selection = False,
		apply_scale_options = 'FBX_SCALE_ALL',
		object_types = {'ARMATURE', 'MESH'},
		bake_anim = True,
		bake_anim_use_all_bones = True,
		bake_anim_force_startend_keying = True,
		path_mode = 'COPY'
	)

def main():
	print(">>>export_fbx.py started...")
	argv = sys.argv
	if "--" not in argv:
		raise RuntimeError("Missing '--' separator")

	idx = argv.index("--") + 1
	rig_path = argv[idx]
	landmarks_path = argv[idx + 1]
	output_path = argv[idx + 2]

	print(f">>>Rig: {rig_path}")
	print(f">>>Landmarks: {landmarks_path}")
	print(f">>>Output: {output_path}")

	if not os.path.exists(landmarks_path):
		raise RuntimeError(f"Landmarks not found: {landmarks_path}")

	landmarks = np.load(landmarks_path)

	clear_scene()
	armature = load_rig(rig_path)
	print(f">>>Armature: {armature.name}")

	rotations = compute_bone_rotations(armature, landmarks, fps = 30)

	#add spine twist
	spine_rotations = compute_spine_rotation(armature, landmarks, fps = 30)
	for bone_name, quats in spine_rotations.items():
		if bone_name in rotations:
			#blend with existing rotation if bone is alr mapped
			rotations[bone_name] = [a @ b for a, b in zip(rotations[bone_name], quats)]
		else:
			rotations[bone_name] = quats

	apply_rotations(armature, rotations, fps = 30)
	apply_root_translation(armature, landmarks, fps = 30)
	export_fbx(output_path)

	if os.path.exists(output_path):
		print(f">>>SUCCESS: {output_path}")
	else:
		print(f">>>WARNING! file missing at {output_path}")

if __name__ == "__main__":
	main()

