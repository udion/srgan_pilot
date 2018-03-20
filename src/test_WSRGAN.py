import argparse
import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable, grad

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models_leaky import Generator, Discriminator, FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=15, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='checkpoints/generator_final.pth', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='checkpoints/discriminator_final.pth', help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='output', help='folder to output results of the model')
parser.add_argument('--modelName', type=str, default='model', help='name of the model (used for creating folder)')

opt = parser.parse_args()
print(opt)

dst_high_real = opt.out+'/'+opt.modelName+'/'+'high_res_real'
dst_high_fake = opt.out+'/'+opt.modelName+'/'+'high_res_fake'
dst_low = opt.out+'/'+opt.modelName+'/'+'low_res'

try:
	os.makedirs(opt.out)
except OSError:
	pass
try:
	os.makedirs(opt.out+'/'+opt.modelName)
except OSError:
	pass
try:
	os.makedirs(dst_high_fake)
	os.makedirs(dst_high_real)
	os.makedirs(dst_low)
except OSError:
	pass

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([transforms.RandomCrop(opt.imageSize*opt.upSampling),
								transforms.ToTensor()])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
								std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
							transforms.Scale(opt.imageSize),
							transforms.ToTensor(),
							transforms.Normalize(mean = [0.485, 0.456, 0.406],
												std = [0.229, 0.224, 0.225])
							])

# Equivalent to un-normalizing ImageNet (for correct visualization)
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

if opt.dataset == 'folder':
	# folder dataset
	dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
elif opt.dataset == 'cifar10':
	dataset = datasets.CIFAR10(root=opt.dataroot, download=True, train=False, transform=transform)
elif opt.dataset == 'cifar100':
	dataset = datasets.CIFAR100(root=opt.dataroot, download=True, train=False, transform=transform)
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
										 shuffle=False, num_workers=int(opt.workers))

G = Generator(16, opt.upSampling)
if opt.generatorWeights != '':
	G.load_state_dict(torch.load(opt.generatorWeights))
print(G)

D = Discriminator()
if opt.discriminatorWeights != '':
	D.load_state_dict(torch.load(opt.discriminatorWeights))
print(D)

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
print(feature_extractor)
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

target_real = Variable(torch.ones(opt.batchSize,1))
target_fake = Variable(torch.zeros(opt.batchSize,1))

one_const = Variable(torch.ones(opt.batchSize, 1))
onebar_const = one_const*-1

# if gpu is to be used
if opt.cuda:
	generator.cuda()
	discriminator.cuda()
	feature_extractor.cuda()
	content_criterion.cuda()
	adversarial_criterion.cuda()
	target_real = target_real.cuda()
	target_fake = target_fake.cuda()

low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

LAMBDA = 1.5
def calc_gp(D, real_data, fake_data):
	eps = torch.rand(opt.batchSize, 1,1,1)
	eps = eps.expand(real_data.size())
	eps = eps.cuda()
	
	interpolated = eps*real_data + (1-eps)*fake_data
	interpolated = interpolated.cuda()
	interpolated = Variable(interpolated, requires_grad=True)
	D_interp = D(interpolated)
	
	gradients = grad(outputs=D_interp, inputs=interpolated, grad_outputs=torch.ones(D_interp.size()).cuda(), 
	create_graph=True, retain_graph=True, only_inputs=True)[0]
	gp = ((gradients.norm(2, dim=1) - 1)**2).mean()*LAMBDA
	return gp

print('Test started...')
mean_generator_content_loss = 0.0
mean_generator_adversarial_loss = 0.0
mean_generator_total_loss = 0.0
mean_discriminator_loss = 0.0

# Set evaluation mode (not training)
G.eval()
D.eval()

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
		high_res_fakev = generator(Variable(low_res).cuda())
	else:
		high_res_realv = Variable(high_res_real)
		high_res_fakev = generator(Variable(low_res))
	
	######### Test discriminator #########
	# with real data
	D_real = D(high_res_realv)
	D_real = D_real.mean()
	# with fake data
	D_fake = D(high_res_fakev)
	D_fake = D_fake.mean()
	# grad penalty term
	gradient_penalty = calc_gp(D, high_res_realv.data, high_res_fakev.data)
	D_loss_wass = D_fake - D_real
	D_loss = D_loss_wass + gradient_penalty
	mean_discriminator_loss += D_loss.data[0]

	######### Test generator #########
	real_features = Variable(feature_extractor(high_res_realv).data)
	fake_features = feature_extractor(high_res_fakev)

	generator_content_loss = content_criterion(high_res_fakev, high_res_realv) + 0.006*content_criterion(fake_features, real_features)
	mean_generator_content_loss += generator_content_loss.data[0]
	G_adversarial_loss = -1*D(high_res_fakev).mean()
	mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

	generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
	mean_generator_total_loss += generator_total_loss.data[0]

	######### Status and display #########
	sys.stdout.write('\r[%d/%d] D_Loss (Wasserstein/GradP/Total): %.4f/%.4f/%.4f, G_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' %
		(i, len(dataloader), D_loss_wass, gradient_penalty, D_loss.data[0], G_content_loss.data[0], G_adversarial_loss.data[0], G_total_loss.data[0]))

	for j in range(opt.batchSize):
		save_image(unnormalize(high_res_real.data[j]), dst_high_real + str(i*opt.batchSize + j) + '.png')
		save_image(unnormalize(high_res_fake.data[j]), dst_high_fake + str(i*opt.batchSize + j) + '.png')
		save_image(unnormalize(low_res[j]), dst_low + str(i*opt.batchSize + j) + '.png')

sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % 
	(i, len(dataloader), mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
		mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))