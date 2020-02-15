import os

import loader
import data_util


def get_training_set(opt):
	print("Loading train data...")
#     import pdb; pdb.set_trace()
	train_sequence = data_util.create_sequence(os.path.join(opt.video_path, 'train'))
	print(f'train_sequence: {train_sequence}')
	train_dataset = loader.VideoLoader(data_dict=train_sequence, 
										   video_resize_x=112,
										   video_resize_y=112,
										   input_name="camera1",
										   output_name="xsens",
										   batch_size=16)
	return train_dataset


def get_validation_set(opt):
	print("Loading validation data...")
	valid_sequence = data_util.create_sequence(os.path.join(opt.video_path, 'test'))
	print(f'valid_sequence: {valid_sequence}')
	valid_dataset = loader.VideoLoader(data_dict=valid_sequence, 
										   video_resize_x=112,
										   video_resize_y=112,
										   input_name="camera1",
										   output_name="xsens",
										   batch_size=16)
	return valid_dataset