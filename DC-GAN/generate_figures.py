from __future__ import print_function
from PIL import Image
import random
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.transforms
import matplotlib.pyplot as plt

import models

model_path = 'results/netG_epoch_50.pth'
#model_path = 'results/netG_epoch_99.pth'
netG = models._netG_1(2, 100, 3,64, 1) # ngpu, nz, nc, ngf, n_extra_layers
netG.cuda()
netG.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

for i in range(1144):
    noise_batch = torch.FloatTensor(1, 100, 1, 1).normal_(0,1)
    noise_batch = Variable(noise_batch)
    fake_batch,_ = netG(noise_batch)
    img_tensor = fake_batch.data.cpu()
    grid = vutils.make_grid(img_tensor, nrow=1, padding=2)
    ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    im = Image.fromarray(ndarr,'RGB')
    im.save('results/dcgan-generate-'+str(i)+'.jpg')