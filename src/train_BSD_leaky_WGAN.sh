#to be done in python3.5 environment with pytorch

python train_leakyrelu_WGAN.py --dataset folder --dataroot ../datasets/BSD/BSDS500/data/images/train --batchSize 4 --imageSize 80 --upSampling 4 --nEpochs 10 --cuda --out ../testBSD_checkpoints/ --modelName X1 --GWeights ../BSD_checkpoints/SRGAN/generator_final.pth --DWeights ../BSD_checkpoints/SRGAN/discriminator_final.pth