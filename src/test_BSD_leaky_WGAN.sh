#to be done in python3.5 environment with pytorch

python test_w_leakyrelu.py --dataset folder --dataroot ../datasets/BSD/BSDS500/data/images/train --batchSize 4 --imageSize 80 --upSampling 4 --cuda --generatorWeights ../BSD_checkpoints/G_final.pth --discriminatorWeights ../BSD_checkpoints/D_final.pth