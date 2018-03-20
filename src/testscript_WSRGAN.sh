#to be done in python3.5 environment with pytorch

python test_WSRGAN.py --dataset folder --dataroot ../datasets/BSD/BSDS500/data/images/test --batchSize 4 --imageSize 80 --upSampling 4 --cuda --generatorWeights ../testBSD_checkpoints/X1/G_final.pth --discriminatorWeights ../testBSD_checkpoints/X1/D_final.pth --out ../outputs --modelName X1