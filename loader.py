import torch 
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import h5py
import numpy as np
import pandas as pd
from pdb import set_trace

# from opts import parse_opts
from image_utils import *

REQ_COLS = ['Prozessschritt', 'Koerperhaltung', 'Kopfhaltung', 'Rumpfdrehung', 'Rumpfneigung', 'Reichweite']

gt = {
	'Koerperhaltung': ['StehenAufr', 'LeichtGeb', 'StehenUeberSchult', 'StehenUeberKopf', 'KnienGeb',
					   'KnienUeberSchult'],
	'Kopfhaltung'   : ['KopfNeutral', 'KopfVornHinten', 'KopfDrehung', 'KopfSeitlich'],
	'Rumpfdrehung'  : ['RumpfdrehKeine', 'RumpfdrehLeicht', 'RumpfdrehMittel', 'RumpfdrehStark'],
	'Rumpfneigung'  : ['RumpfneigKeine', 'RumpfneigLeicht'],
	'Reichweite'    : ['ReichwNah']
}

unq_1 = np.linspace(0, 1, len(gt['Koerperhaltung'])+1)
unq_2 = np.linspace(0, 1, len(gt['Kopfhaltung'])+1)
unq_3 = np.linspace(0, 1, len(gt['Rumpfdrehung'])+1)
unq_4 = np.linspace(0, 1, len(gt['Rumpfneigung'])+1)
unq_5 = np.linspace(0, 1, len(gt['Reichweite'])+1)


class VideoLoader(Dataset):
	def __init__(self, data_dict, video_resize_x, video_resize_y, input_name, output_name, batch_size):
		self.data_dict= data_dict
		self.data_set = list(data_dict.keys())
		self.total_data_set_size = len(self.data_set)
		self.video_resize_x = video_resize_x
		self.video_resize_y = video_resize_y
		self.input_name = input_name
		self.output_name = output_name
		self.batch_size = batch_size
		self.buffer_ptr = 0
		self.current_video_idx = 0
		self.current_total_buffer = 0
		self.channels_first = True
		self.video_frame_buffer = []
		self.excel_data_buffer = []

	def load_buffer(self, idx):
		if  self.current_video_idx < self.total_data_set_size and self.buffer_ptr >= self.current_total_buffer:
			print(f"loading video: {self.current_video_idx}")
			# new_video_frame_buffer, new_sensor_data_buffer = self.item_loader(self.data_set[self.current_video_idx])
			self.video_frame_buffer, self.excel_data_buffer = self.item_loader(self.data_set[self.current_video_idx])
			# print(f"Total number of clips: {len(video_frame_buffer)}")
			self.current_total_buffer = self.video_frame_buffer.shape[0] # or self.excel_data_buffer.shape[0]
			self.current_video_idx += 1
			self.buffer_ptr = 0

	def __getitem__(self, idx):
		print(f'Index: {idx}')
		self.load_buffer(idx)
		if self.buffer_ptr < self.current_total_buffer:
			# inputs  = torch.from_numpy(video_frame_buffer) #.type(torch.FloatTensor)
			inputs = self.video_frame_buffer[self.buffer_ptr]
			if inputs.shape[0] > self.batch_size:
				inputs = inputs[:self.batch_size, :, :, :]
			elif inputs.shape[0] < self.batch_size:
				f, c, h, w = inputs.shape
				pad = self.batch_size - inputs.shape[0]
				zeros = np.zeros((pad, c, h, w), dtype=np.float32)
				inputs = np.concatenate((inputs, zeros))
			if self.channels_first:
				inputs = np.rollaxis(inputs, 3, 0)
			inputs  = torch.tensor(inputs, dtype=torch.float32)
			print('shape of input data: ', inputs.shape)
			# outputs = torch.from_numpy(excel_data_buffer) #.type(torch.FloatTensor)
			outputs = torch.tensor(self.excel_data_buffer[self.buffer_ptr].astype(np.float32), dtype=torch.float32) #.type(torch.FloatTensor)
			print('shape of label data: ', outputs.shape)
			self.buffer_ptr += 1
			return inputs, outputs
		return None, None

	def __len__(self):
		return self.total_data_set_size
		#return int(np.ceil(len(self.data_set) / float(self.batch_size)))
		
	def get_excel_data(self, excel_path):        
		print(f'Read annotation xlsx: {excel_path}')
		df = pd.read_excel(excel_path, sheet_name='EAB2 - Autom. Zusammenfassen', skiprows=1)
		positions = []
		for pos0, pos1, pos2, pos3, pos4, pos5 in zip(df[REQ_COLS[0]], df[REQ_COLS[1]], df[REQ_COLS[2]], df[REQ_COLS[3]], df[REQ_COLS[4]], df[REQ_COLS[5]]):
			if len(pos0.split('-')) == 1:
				start = int(pos0.strip().split('Frame')[-1])
				end = start + 1
			elif len(pos0.split('-')) == 2:
				start, end = pos0.strip().split('-')
				start, end = int(start.split('Frame')[-1].strip()), int(end.split('Frame')[-1].strip())
			gt1 = unq_1[gt[REQ_COLS[1]].index(pos1.strip())+1]
			gt2 = unq_2[gt[REQ_COLS[2]].index(pos2.strip())+1]
			gt3 = unq_3[gt[REQ_COLS[3]].index(pos3.strip())+1]
			gt4 = unq_4[gt[REQ_COLS[4]].index(pos4.strip())+1]
			gt5 = unq_5[gt[REQ_COLS[5]].index(pos5.strip())+1]
			positions.append([np.array([[start, end], gt1, gt2, gt3, gt4, gt5])])
		print('Shape of sensor data:', np.array(positions).shape)
		return np.array(positions)

	def get_sensor_data(self, h5_path):
		with h5py.File(h5_path, 'r') as hf:
			angles=hf['jointAngle'][:]
		sensor_data = torch.from_numpy(angles)
		return sensor_data

	def get_video_frames(self, video_path):
		print('Processing video file: ', video_path)
		cap = cv2.VideoCapture(video_path)
		# fps = cap.get(cv2.CAP_PROP_FPS)
		# timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
		frames = []
		frame_count = 1
		while cap.isOpened():
			frame_exists, curr_frame = cap.read()
			if not frame_exists:
				break
			curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
			curr_frame = resize_image(curr_frame, (self.video_resize_x, self.video_resize_y))
			curr_frame = normalize_image(curr_frame)            
			frames.append(curr_frame)
			print(f'Frames processed: {frame_count}', end='\r')
			frame_count += 1
		cap.release()
		print('')
		frames = np.array(frames)
		print(f'Output shape of video: {frames.shape}')
		return frames
	
	def get_batch(self, clip_path, frames, sensor_data):
		# name = ".".join(clip_path.split('.')[:-1])
		# frame_height, frame_width = frames[0].shape[:2]
		# fourcc = cv2.VideoWriter_fourcc(*'XVID')
		print('Processing video clips and sensor data...')
# 		import pdb; pdb.set_trace()
		video_clips = []
		sensor_clips = []
		for i, clip_window in enumerate(sensor_data):
			start, end = clip_window[0][0]
			# out = cv2.VideoWriter(os.path.join(SAVE_CLIPS_FOLDER, f'{name}_g{v_id}_c{count}.avi'), fourcc, 20.0, (frame_width, frame_height))
			# print(f'start: {start}, end: {end}')
			if frames[start-1:end+1, :, :, :].shape[0] != 0:
				video_clips.append(frames[start-1:end+1, :, :, :])
				sensor_clips.append(clip_window[0][1:])
			# out.release()
		video_clips = np.array(video_clips) #.astype(np.float32)
		sensor_clips = np.array(sensor_clips) #.astype(np.float32)
		print(f'Output shape of clips: {video_clips.shape}')
		print(f'Output shape of sensor data: {sensor_clips.shape}')
		return video_clips, sensor_clips

	def read_data_sample(self, video_path, excel_data_path):
		# sensor_data = self.get_sensor_data(sensor_data_path)
		excel_data = self.get_excel_data(excel_data_path)
		video_data = self.get_video_frames(video_path)
		clip_data, excel_data = self.get_batch(video_path, video_data, excel_data)
		
		# video_data_len = len(video_data)
		# sensor_data_len = len(sensor_data)
		# excel_data = len(excel_path)
		
		# data_len = min(sensor_data_len, video_data_len)
		# data_len = min(sensor_data_len, excel_data)
		# return video_data[:data_len], sensor_data[:data_len]
		return clip_data, excel_data

	def item_loader(self, item):
		_input_path = self.data_dict[item][self.input_name]
		# _output_path = self.data_dict[item][self.output_name]
		file = os.path.basename(_input_path)
		_excel_path = os.path.join(os.path.dirname(_input_path), ".".join(file.split('.')[:-1])+'.xlsx')
		# return self.read_data_sample(_input_path, _output_path)
		return self.read_data_sample(_input_path, _excel_path)


# val = {'Participant_542_Setup_A_Seq_5_Trial_3': {'camera1': 'video/test/Participant_542_Setup_A_Seq_5_Trial_3.camera1.mp4'}}
# # opt = parse_opts()
# train_dataset = VideoLoader(data_dict=val, 
# 										   video_resize_x=112,
# 										   video_resize_y=112,
# 										   input_name="camera1",
# 										   output_name="xsens",
# 										   batch_size=16)

# # print(train_dataset.item_loader('Participant_541_Setup_A_Seq_5_Trial_3'))
# # print(train_dataset.__getitem__())
# for i, (image, label) in enumerate(train_dataset):
# 	if image is None:
# 		break
# 	pass