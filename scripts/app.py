#app.py

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import sys
import os
import traceback

#path setup handles dev mode and pyinstaller
if getattr(sys, 'frozen', False):
	BASE = sys._MEIPASS
else:
	BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(BASE, "scripts"))
import run


class App(ctk.CTk):
	def __init__(self):
		super().__init__()
		self.title("MP4 to FBX")
		self.geometry("600x460")

		self.video_path = ""
		self.output_folder = ""

		ctk.CTkLabel(self, text = "MP4 to FBX Animation Converter", font = ("Arial", 20)).pack(pady = 20)

		#video selection
		self.btn_video = ctk.CTkButton(self, text = "Select MP4 Video", command = self.select_video)
		self.btn_video.pack(pady = 10)

		self.lbl_video = ctk.CTkLabel(self, text = "No Video Selected...")
		self.lbl_video.pack()

		#output folder
		self.btn_output = ctk.CTkButton(self, text = "Select Output Folder", command = self.select_output)
		self.btn_output.pack(pady = 10)

		self.lbl_output = ctk.CTkLabel(self, text = "No Folder Selected...")
		self.lbl_output.pack()

		#preview for final output path
		self.lbl_preview = ctk.CTkLabel(self, text = "", text_color = "gray")
		self.lbl_preview.pack(pady = (4, 0))

		#process btn
		self.btn_process = ctk.CTkButton(self, text = "PROCESS", command = self.start_processing, fg_color = "green", height = 40)
		self.btn_process.pack(pady = 24)

		#status
		self.lbl_status = ctk.CTkLabel(self, text = "Ready")
		self.lbl_status.pack()

	def update_preview(self):
		if self.video_path and self.output_folder:
			name = os.path.splitext(os.path.basename(self.video_path))[0]
			preview = os.path.join(self.output_folder, name + ".fbx")
			self.lbl_preview.configure(text = "Will save to: " + preview)
		else:
			self.lbl_preview.configure(text = "")

	def select_video(self):
		path = filedialog.askopenfilename(filetypes = [("MP4 files", "*.mp4")])
		if path:
			self.video_path = path
			self.lbl_video.configure(text = os.path.basename(path))
			self.update_preview()

	def select_output(self):
		folder = filedialog.askdirectory()
		if folder:
			self.output_folder = folder
			self.lbl_output.configure(text = folder)
			self.update_preview()

	def start_processing(self):
		if not self.video_path or not self.output_folder:
			messagebox.showerror("Error", "Select BOTH a video and an output folder.")
			return

		name = os.path.splitext(os.path.basename(self.video_path))[0]
		output_fbx = os.path.join(self.output_folder, name + ".fbx")

		self.btn_process.configure(state = "disabled", text = "Processing...")
		self.lbl_status.configure(text = "Working...")

		thread = threading.Thread(target = self.run_pipeline, args = (output_fbx,))
		thread.start()

	def run_pipeline(self, output_fbx):
		try:
			run.main(self.video_path, output_fbx)

			if os.path.exists(output_fbx):
				self.lbl_status.configure(text = "Done!")
				messagebox.showinfo("Success", "FBX Exported Successfully!\n\n" + output_fbx)
			else:
				self.lbl_status.configure(text = "Failed - see console")
				messagebox.showerror("Failed", "Pipeline finished but no output was created.")
		except Exception as e:
			print("--- PIPELINE ERROR ---")
			traceback.print_exc()
			self.lbl_status.configure(text = "Error: " + str(e))
			messagebox.showerror("Error", str(e))
		finally:
			self.btn_process.configure(state = "normal", text = "PROCESS")


if __name__ == "__main__":
	app = App()
	app.mainloop()