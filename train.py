import torch
# import torch.nn as nn
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy
# from models.convlstm import *
# from models.slowfast import SlowFast

# slowfast = SlowFast(layers=[3, 4, 6, 3], class_num=5, shortcut_type='B', dropout=0.5, alpha=8, beta=0.125)
# slowfast = slowfast.to('cuda')

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
				epoch_logger, batch_logger):
	print('train at epoch {}'.format(epoch))

	model.train()

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracies = AverageMeter()
#     mse = nn.MSELoss()
	end_time = time.time()
# 	import pdb; pdb.set_trace()    
	for i, (inputs, targets) in enumerate(data_loader):
		if inputs is None:
			break
		data_time.update(time.time() - end_time)
		if not opt.no_cuda:
			targets = targets.cuda()
		inputs = Variable(inputs)
		targets = Variable(targets)
		# expand dims
		inputs = inputs.expand(1, -1, -1, -1, -1)
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		acc = calculate_accuracy(outputs, targets)
		losses.update(loss.data, inputs.size(0))
		accuracies.update(acc, inputs.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end_time)
		end_time = time.time()

		batch_logger.log({
			'epoch': epoch,
			'batch': i + 1,
			'iter': (epoch - 1) * len(data_loader) + (i + 1),
			'loss': losses.val,
			'acc': accuracies.val,
			'lr': optimizer.param_groups[0]['lr']
		})

		print('Epoch: [{0}][{1}/{2}]\t'
			  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
			  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
			  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
			  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
				  epoch,
				  i + 1,
				  len(data_loader),
				  batch_time=batch_time,
				  data_time=data_time,
				  loss=losses,
				  acc=accuracies))

	epoch_logger.log({
		'epoch': epoch,
		'loss': losses.avg,
		'acc': accuracies.avg,
		'lr': optimizer.param_groups[0]['lr']
	})

	if epoch % opt.checkpoint == 0:
		save_file_path = os.path.join(opt.result_path,
									  'save_{}.pth'.format(epoch))
		states = {
			'epoch': epoch + 1,
			'arch': opt.arch,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}
		torch.save(states, save_file_path)
