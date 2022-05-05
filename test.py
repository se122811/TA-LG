from os import makedirs
from torch.utils.data.dataloader import DataLoader

import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets import *
from models import *

makedirs("/data/yunhoe/images_test", exist_ok=True)

cuda = True if torch.cuda.is_available() else False

dataset = "img_masked"
generator = Generator()
generator.load_state_dict(torch.load('/data/yunhoe/pretrained_pth/generator_196.pth'))

transforms_ = [
    transforms.Resize((128, 128), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

if cuda:
    generator.cuda()

dataloader = DataLoader(
    ImageDataset(f"../../data/img_test", transforms_=transforms_, mode='test'),
    batch_size=20,
    shuffle=True,
    num_workers=4,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def save_sample():
    samples, masked_samples, i = next(iter(dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()  # Upper-left coordinate of mask
    # Generate inpainted image
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i : i + 64, i : i + 64] = gen_mask
    # filled_samples[:, :, i + 16 : i + 80, i : i + 64] = gen_mask
    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, "/data/yunhoe/images_test/.png", nrow=6, normalize=True)

save_sample()