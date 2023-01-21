# CMU 16-726 Learning-Based Image Synthesis / Spring 2021, Assignment 3
#
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# This is the main training file for the first part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

# NEW VISUALIZATION PACKAGES ***********
# Visualization
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imsave
# %matplotlib inline


# original imports
import argparse
import os
import warnings

import imageio

warnings.filterwarnings("ignore")

# Numpy & Scipy imports
import numpy as np

# Torch imports
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Variable 

# Local imports
import utils
from data_loader import get_data_loader
from models import DCGenerator, DCDiscriminator


SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)



# NEW VISUALIZATION FUNCTION ***************

def report(D_loss_records, G_loss_records):
  rows = 1
  cols = 2

  fig, axis = plt.subplots(nrows = rows, 
                          ncols = cols, 
                          squeeze = True,
                          gridspec_kw = {'wspace':0, 'hspace':0})

  fig.set_figheight(10)
  fig.set_figwidth(40)

  #axis[0].imshow(img, aspect='auto',cmap='gray')
  axis[0].axis(False)
  axis[1].plot(D_loss_records, c= 'red', label='D Loss', linewidth= 1)
  axis[1].plot(G_loss_records, c= 'blue',  label='G Loss', linewidth= 1)
  axis[1].set_title("D and G Loss", fontsize='x-large')
  axis[1].set_xlabel("Epoch", fontsize='x-large')
  axis[1].legend(loc='upper right', shadow=False, fontsize='x-large')
  plt.savefig('output/plot.png')
#   plt.show()




def print_models(G, D):
    """Prints model information for the generators and discriminators.
    """
    print("                    G                  ")
    print("---------------------------------------")
    print(G)
    print("---------------------------------------")

    print("                    D                  ")
    print("---------------------------------------")
    print(D)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """
    G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.conv_dim)
    D = DCDiscriminator(conv_dim=opts.conv_dim)

    print_models(G, D)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')

    return G, D


def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h*nrows, cell_w*ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = array[i*ncols+j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result


def checkpoint(iteration, G, D, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G_iter%d.pkl' % iteration)
    D_path = os.path.join(opts.checkpoint_dir, 'D_iter%d.pkl' % iteration)
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)


def save_samples(G, fixed_noise, iteration, opts):
    generated_images = G(fixed_noise)
    generated_images = utils.to_data(generated_images)

    grid = create_image_grid(generated_images)

    # merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}.png'.format(iteration))
    imageio.imwrite(path, grid)
    print('Saved {}'.format(path))


def save_images(images, iteration, opts, name):
    grid = create_image_grid(utils.to_data(images))

    path = os.path.join(opts.sample_dir, '{:s}-{:06d}.png'.format(name, iteration))
    imageio.imwrite(path, grid)
    print('Saved {}'.format(path))


def sample_noise(dim):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return utils.to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def training_loop(train_dataloader, opts):
    """Runs the training loop.
        * Saves checkpoints every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G, D = create_model(opts)

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(G.parameters(), opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(D.parameters(), opts.lr, [opts.beta1, opts.beta2])

    # Generate fixed noise for sampling from the generator
    fixed_noise = sample_noise(opts.noise_size)  # batch_size x noise_size x 1 x 1

    iteration = 1

    total_train_iters = opts.num_epochs * len(train_dataloader)

    # LOSS RECORDS - For Model Performance Visualization ************
    D_loss_records = []
    G_loss_records = []

    # original

    for epoch in range(opts.num_epochs):


        for batch in train_dataloader:

            real_images, labels = batch
            real_images, labels = utils.to_var(real_images), utils.to_var(labels).long().squeeze()

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################

            d_optimizer.zero_grad()

            # Good references & guides for calculating loss function
            # guide 1: https://ltquesada.medium.com/your-first-cyclegan-using-pytorch-85546dfe6317
            # guide 2: https://machinelearningmastery.com/least-squares-generative-adversarial-network/
            # guide 3: https://wiseodd.github.io/techblog/2017/03/02/least-squares-gan/

            # FILL THIS IN
            # 1. Compute the discriminator loss on real images
            D_real = D(real_images)
            D_real_loss = torch.mean((D_real - 1)**2)

            # 2. Sample noise
            noise = sample_noise(opts.noise_size)


            # 3. Generate fake images from the noise
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images   
            D_fake = D(fake_images)
            D_fake_loss = torch.mean(D_fake**2)

            # Add real loss and fake loss together
            # Below code learned and adapted from guide 3
            D_total_loss = 0.5 * (torch.mean((D_real - 1)**2) + torch.mean(D_fake**2))

            if iteration % 2 == 0:
                D_total_loss.backward()
                d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################

            g_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Sample noise
            noise = sample_noise(opts.noise_size)

            # 2. Generate fake images from the noise
            fake_images = G(noise)

            # 3. Compute the generator loss
            # Below code learned and adapted from guide 3
            D_fake = D(fake_images)
            G_loss = 0.5 * torch.mean((D_fake - 1)**2)

            G_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                       iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))
            # todo: add fake loss, real loss, G loss to tensorboard


            # Save the generated samples
            if iteration % opts.sample_every == 0:
                save_samples(G, fixed_noise, iteration, opts)
                save_images(real_images, iteration, opts, 'real')

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            # original
            iteration += 1

        # LOSS RECORDS - Save Every Other Iteration **********
        if epoch % 1 == 0:
            D_loss_records.append(D_total_loss.item())
            G_loss_records.append(G_loss.item())

        # DISPLAY VISUALIZATION
        if epoch + 1  == opts.num_epochs:
        # if iteration % opts.checkpoint_every == 0:
            report(D_loss_records,G_loss_records)




def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create a dataloader for the training images
    dataloader = get_data_loader(opts.data, opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    training_loop(dataloader, opts)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=64, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=32)
    parser.add_argument('--noise_size', type=int, default=100)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--data', type=str, default='cat/grumpifyBprocessed', help='Choose the type of emojis to generate.')
    parser.add_argument('--data_aug', type=str, default='deluxe', help='data augmentation diff / basic / deluxe')
    parser.add_argument('--ext', type=str, default='*.png', help='Choose the type of emojis to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='./vanilla')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=400)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size
    opts.sample_dir = os.path.join('output/', opts.sample_dir,
                                   '%s_%s' % (os.path.basename(opts.data), opts.data_aug))
    if os.path.exists(opts.sample_dir):
        cmd = 'rm %s/*' % opts.sample_dir
        os.system(cmd)
    logger = SummaryWriter(opts.sample_dir)
    print(opts)
    main(opts)

# python3 vanilla_gan.py --num_epochs=100 --data_aug=basic


