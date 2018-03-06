import _pickle
import numpy as np
import random
import os, sys
import PIL
from PIL import Image
random.seed(1) # set a seed so that the results are consistent

#creating dataset folder
datasets_root = '../datasets'
dataset_name = 'cifar10'
try:
	os.makedirs(datasets_root)
	os.makedirs(datasets_root+'/'+dataset_name)
	os.makedirs(datasets_root+'/'+dataset_name+'/train')
	os.makedirs(datasets_root+'/'+dataset_name+'/test')
except OSError:
	pass
for i in range(10):
	try:
		os.mkdir(datasets_root+'/'+dataset_name+'/train/{}'.format(i))
	except OSError:
		pass
	try:
		os.mkdir(datasets_root+'/'+dataset_name+'/test/{}'.format(i))
	except OSError:
		print('I tried')
		pass

class_count = np.zeros(10, dtype='int16')

def load_batch(batch_num, root_path):
	path = root_path
	if batch_num == -1:
		file = 'test_batch'
	else:
		file = 'data_batch_{}'.format(batch_num)

	f = open(path+'/'+file, 'rb')
	dict = _pickle.load(f, encoding='latin1')
	images = dict['data']
	#images = np.reshape(images, (10000, 3, 32, 32))
	labels = dict['labels']
	imagearray = np.array(images)   #   (10000, 3072)
	labelarray = np.array(labels)   #   (10000,)
	
	return imagearray, labelarray

def recreate_databatch(batch_num, root_path):
	batch_imgarr, batch_labels = load_batch(batch_num, root_path)
	orig_image_sz = 32
	n_channels = 3
	orig_image_sz2 = orig_image_sz*orig_image_sz
	num_in_batch = batch_imgarr.shape[0]
	batch_imgs = np.reshape(batch_imgarr, (num_in_batch, orig_image_sz, orig_image_sz, n_channels))
	for i in range(num_in_batch):
		label = batch_labels[i]
		npimg = np.empty((32,32,3))
		imgarr = batch_imgarr[i]
		r_ch = imgarr[0:1024]
		g_ch = imgarr[1024:2048]
		b_ch = imgarr[2048:3072]
		npimg[:,:,0] = np.reshape(r_ch, (32,32))
		npimg[:,:,1] = np.reshape(g_ch, (32,32))
		npimg[:,:,2] = np.reshape(b_ch, (32,32))
		img = Image.fromarray(npimg.astype('uint8'), 'RGB')
		if batch_num == -1:
			img.save(datasets_root+'/'+dataset_name+'/test/{}'.format(label)+'/{}.jpg'.format(class_count[label]))
		else:
			img.save(datasets_root+'/'+dataset_name+'/train/{}'.format(label)+'/{}.jpg'.format(class_count[label]))
		class_count[label] += 1

def recreate_dataset(mode):
	if mode=='train':
		recreate_databatch(1, './cifar-10-batches-py')
		recreate_databatch(2, './cifar-10-batches-py')
		recreate_databatch(3, './cifar-10-batches-py')
		recreate_databatch(4, './cifar-10-batches-py')
		recreate_databatch(5, './cifar-10-batches-py')
	elif mode=='test':
		recreate_databatch(-1, './cifar-10-batches-py')
	print('done')


recreate_dataset('test')