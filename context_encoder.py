from os import makedirs
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets import *
from models import *

makedirs("/data/yunhoe/images_input_256", exist_ok=True)
makedirs("/data/yunhoe/pretrained_pth_input_256", exist_ok=True)

cuda = True if torch.cuda.is_available() else False

dataset_name = "img_align_celeba"
learning_rate = 0.0002
epoch = 500

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(64 / 2 ** 2), int(64 / 2 ** 2)
patch = (1, patch_h, patch_w)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# a_loss: adversarial loss, p_loss: pixelwise loss
a_loss, p_loss = nn.MSELoss(), nn.L1Loss()

generator, discriminator = Generator(), Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    a_loss.cuda()
    p_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Dataset loader
transforms_ = [
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset("../../data/%s" % dataset_name, transforms_=transforms_),
    batch_size=128,
    shuffle=True,
    num_workers=4,
)
test_dataloader = DataLoader(
    ImageDataset("../../data/%s" % dataset_name, transforms_=transforms_, mode="val"),
    batch_size=128,
    shuffle=True,
    num_workers=1,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Scheduler
# scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_G, step_size=100, gamma=0.9)
# scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_D, step_size=100, gamma=0.9)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def save_sample(batches_done):
    samples, masked_samples, i = next(iter(test_dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()  # Upper-left coordinate of mask
    # Generate inpainted image
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i : i + 64, i : i + 64] = gen_mask
    # Save sample
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, "/data/yunhoe/images_input_256/%d.png" % batches_done, nrow=6, normalize=True)

# ----------
#  Training
# ----------

for ith in tqdm(range(epoch)):
    for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_parts = generator(masked_imgs)

        # Adversarial and pixelwise loss
        g_adv = a_loss(discriminator(gen_parts), valid)
        g_pixel = p_loss(gen_parts, masked_parts)
        # Total loss
        g_loss = 0.001 * g_adv + 0.999 * g_pixel

        g_loss.backward()
        optimizer_G.step()
        # scheduler_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = a_loss(discriminator(masked_parts), valid)
        fake_loss = a_loss(discriminator(gen_parts.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
        # scheduler_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
            % (ith, epoch, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
        )

        # Generate sample at sample interval
        batches_done = ith * len(dataloader) + i
        if batches_done % 100 == 0:
            save_sample(batches_done)
            print()
    

    if ith % 5 == 0:
        torch.save(generator.state_dict(), '/data/yunhoe/pretrained_pth_input_256/generator_{}.pth'.format(ith + 1))
        torch.save(discriminator.state_dict(), '/data/yunhoe/pretrained_pth_input_256/discriminator_{}.pth'.format(ith + 1))