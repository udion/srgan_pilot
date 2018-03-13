import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable, grad

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tensorboard_logger import configure, log_value

from models_leaky import Generator, Discriminator, FeatureExtractor
from utils import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=15, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--GLR', type=float, default=0.0001, help='learning rate for G')
parser.add_argument('--DLR', type=float, default=0.0001, help='learning rate for D')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--GWeights', type=str, default='', help="path to G weights (to continue training)")
parser.add_argument('--DWeights', type=str, default='', help="path to D weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)

try:
	os.makedirs(opt.out)
except OSError:
	pass

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([transforms.RandomCrop(opt.imageSize*opt.upSampling),
								transforms.ToTensor()])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
								std = [0.229, 0.224, 0.225])

#this will help generate the smaller resolution version of the images, first convert it 
#to 0 1 scale then add gaussian noise then scale and (then redo the standard normalisation)
scale = transforms.Compose([transforms.ToPILImage(),
							transforms.Scale(opt.imageSize),
							transforms.ToTensor(),
							transforms.Normalize(mean = [0.485, 0.456, 0.406],
												std = [0.229, 0.224, 0.225])
							])

if opt.dataset == 'folder':
	# folder dataset
	dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
elif opt.dataset == 'cifar10':
	dataset = datasets.CIFAR10(root=opt.dataroot, train=True, download=True, transform=transform)
elif opt.dataset == 'cifar100':
	dataset = datasets.CIFAR100(root=opt.dataroot, train=True, download=True, transform=transform)
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
										 shuffle=True, num_workers=int(opt.workers))

################# initialising the model ########################
G = Generator(16, opt.upSampling)
if opt.GWeights != '':
	G.load_state_dict(torch.load(opt.GWeights))
print(G)

D = Discriminator()
if opt.DWeights != '':
	D.load_state_dict(torch.load(opt.DWeights))
print(D)
#################################################################

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
print(feature_extractor)
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

ones_const = Variable(torch.ones(opt.batchSize, 1))

# if gpu is to be used
if opt.cuda:
	G.cuda()
	D.cuda()
	feature_extractor.cuda()
	content_criterion.cuda()
	adversarial_criterion.cuda()
	ones_const = ones_const.cuda()

optim_G = optim.Adam(G.parameters(), lr=opt.GLR)
optim_D = optim.Adam(D.parameters(), lr=opt.DLR)

configure('logs/'+ opt.dataset + '-' + str(opt.batchSize)+ str(opt.dataroot[1]) + '-' + str(opt.GLR) + '-' + str(opt.DLR), flush_secs=5)
print('I configured .. ')
# visualizer = Visualizer(image_size=opt.imageSize*opt.upSampling)

low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

# Pre-train G using raw MSE loss
print('G pre-training')
for epoch in range(2):
	mean_G_content_loss = 0.0

	for i, data in enumerate(dataloader):
		# Generate data
		high_res_real, _ = data

		# Downsample images to low resolution
		for j in range(opt.batchSize):
			low_res[j] = scale(high_res_real[j])
			high_res_real[j] = normalize(high_res_real[j])

		# Generate real and fake inputs
		if opt.cuda:
			high_res_real = Variable(high_res_real.cuda())
			high_res_fake = G(Variable(low_res).cuda())
		else:
			high_res_real = Variable(high_res_real)
			high_res_fake = G(Variable(low_res))

		######### Train G #########
		G.zero_grad()

		G_content_loss = content_criterion(high_res_fake, high_res_real)
		mean_G_content_loss += G_content_loss.data[0]

		G_content_loss.backward()
		optim_G.step()

		######### Status and display #########
		sys.stdout.write('\r[%d/%d][%d/%d] G_MSE_Loss: %.4f' % (epoch, 2, i, len(dataloader), G_content_loss.data[0]))
		# visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.cpu().data)

	sys.stdout.write('\r[%d/%d][%d/%d] G_MSE_Loss: %.4f\n' % (epoch, 2, i, len(dataloader), mean_G_content_loss/len(dataloader)))
	log_value('G_mse_loss', mean_G_content_loss/len(dataloader), epoch)

# Do checkpointing
torch.save(G.state_dict(), '%s/G_pretrain.pth' % opt.out)

# SRGAN training
optim_G = optim.Adam(G.parameters(), lr=opt.GLR*0.1)
optim_D = optim.Adam(D.parameters(), lr=opt.DLR*0.1)

############# for WGAN training ##################################
N_discri = 5 #D trained 5 times per training of G
LAMBDA = 0.10
def calc_gp(D, real_data, fake_data):
	eps = torch.rand(opt.batchSize, 1,1,1)
	# print(opt.batchSize, real_data.size()[0], real_data.size()[1], real_data.size()[2], real_data.size()[3])
	eps = eps.expand(real_data.size())
	eps = eps.cuda()
	
	interpolated = eps*real_data + (1-eps)*fake_data
	interpolated = interpolated.cuda()
	interpolated = Variable(interpolated, requires_grad=True)
	D_interp = D(interpolated)
	
	gradients = grad(outputs=D_interp, inputs=interpolated, grad_outputs=torch.ones(D_interp.size()).cuda()
						,create_graph=True, retain_graph=True, only_inputs=True)[0]
	gp = ((gradients.norm(2, dim=1) - 1)**2).mean()*LAMBDA
	return gp
###################################################################

print('SRGAN training using WGAN method ...')
for epoch in range(opt.nEpochs):
	mean_G_content_loss = 0.0
	mean_G_adversarial_loss = 0.0
	mean_G_total_loss = 0.0
	mean_D_loss = 0.0

	for i, data in enumerate(dataloader):
		# Generate data
		high_res_real, _ = data

		# Downsample images to low resolution
		for j in range(opt.batchSize):
			low_res[j] = scale(high_res_real[j])
			high_res_real[j] = normalize(high_res_real[j])

		# Generate real and fake inputs
		if opt.cuda:
			high_res_realv = Variable(high_res_real.cuda())
			high_res_fakev = G(Variable(low_res).cuda())
			target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda()
			target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda()
		else:
			high_res_realv = Variable(high_res_real)
			high_res_fakev = G(Variable(low_res))
			target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7)
			target_fake = Variable(torch.rand(opt.batchSize,1)*0.3)

		######### Train D #########
		for p in D.parameters():
			p.requires_grad = True
		for nd in range(N_discri):
			D.zero_grad()
			# with real data
			high_res_realv = Variable(high_res_real.cuda())
			high_res_fakev = G(Variable(low_res).cuda())
			D_real = D(high_res_realv)
			D_real = D_real.mean()
			# with fake data
			D_fake = D(high_res_fakev)
			D_fake = D_fake.mean()
			# grad penalty term
			gradient_penalty = calc_gp(D, high_res_realv.data, high_res_fakev.data)
		   
			D_loss_wass = D_fake - D_real
			D_loss = D_loss_wass + gradient_penalty
			# D_loss = adversarial_criterion(D(high_res_real), target_real) + \
			#                     adversarial_criterion(D(Variable(high_res_fake.data)), target_fake)
			mean_D_loss += D_loss.data[0]
			# print('look here ', mean_D_loss)
			D_loss.backward()
			optim_D.step()

		######### Train G #########
		high_res_realv = Variable(high_res_real.cuda())
		high_res_fakev = G(Variable(low_res).cuda())
		for p in D.parameters():
			p.requires_grad = False  # to avoid computation
		G.zero_grad()

		real_features = Variable(feature_extractor(high_res_realv).data)
		fake_features = feature_extractor(high_res_fakev)

		G_content_loss = content_criterion(high_res_fakev, high_res_realv) + 0.006*content_criterion(fake_features, real_features)
		mean_G_content_loss += G_content_loss.data[0]
		G_adversarial_loss = adversarial_criterion(D(high_res_fakev), ones_const)
		mean_G_adversarial_loss += G_adversarial_loss.data[0]

		G_total_loss = G_content_loss + 1e-3*G_adversarial_loss
		mean_G_total_loss += G_total_loss.data[0]
		
		G_total_loss.backward()
		optim_G.step()   
		
		######### Status and display #########
		sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f, D_loss_wass: %.4f, gradient penalty: %.4f, G_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' %
			(epoch, opt.nEpochs, i, len(dataloader), D_loss.data[0], D_loss_wass, gradient_penalty, G_content_loss.data[0], G_adversarial_loss.data[0], G_total_loss.data[0]))
		# visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.cpu().data)
	
	sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f, D_loss_wass: %.4f, gradient penalty: %.4f, \n G_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' %
		(epoch, opt.nEpochs, i, len(dataloader), D_loss.data[0], D_loss_wass, gradient_penalty, G_content_loss.data[0], G_adversarial_loss.data[0], G_total_loss.data[0]))
	# sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f G_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch, opt.nEpochs, i, len(dataloader),
	# mean_D_loss/len(dataloader), mean_G_content_loss/len(dataloader), 
	# mean_G_adversarial_loss/len(dataloader), mean_G_total_loss/len(dataloader)))

	log_value('G_content_loss', mean_G_content_loss/len(dataloader), epoch)
	log_value('G_adversarial_loss', mean_G_adversarial_loss/len(dataloader), epoch)
	log_value('G_total_loss', mean_G_total_loss/len(dataloader), epoch)
	log_value('D_loss', mean_D_loss/len(dataloader), epoch)

	# Do checkpointing
	torch.save(G.state_dict(), '%s/G_final.pth' % opt.out)
	torch.save(D.state_dict(), '%s/D_final.pth' % opt.out)

# Avoid closing
while True:
	pass
