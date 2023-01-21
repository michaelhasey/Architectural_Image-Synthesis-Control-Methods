
# Code learned and adapted from Yufei's master file


from __future__ import print_function

import argparse
import os
import os.path as osp
import numpy as np

from LBFGS import FullBatchLBFGS

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import torchvision.utils as vutils
from torchvision.models import vgg19

from dataloader import get_data_loader


def build_model(name):
    if name.startswith('vanilla'):
        z_dim = 100
        model_path = 'pretrained/%s.ckpt' % name
        # pretrain = torch.load(model_path)
        pretrain =  torch.load(model_path, map_location='cuda:0')
        from vanilla.models import DCGenerator
        model = DCGenerator(z_dim, 32, 'instance')
        model.load_state_dict(pretrain)

    elif name == 'stylegan':
        model_path = 'pretrained/%s.ckpt' % name
        import sys
        sys.path.insert(0, 'stylegan')
        from stylegan import dnnlib, legacy
        with dnnlib.util.open_url(model_path) as f:
            model = legacy.load_network_pkl(f)['G_ema']
            z_dim = model.z_dim
    else:
         return NotImplementedError('model [%s] is not implemented', name)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, z_dim


class Wrapper(nn.Module):
    """The wrapper helps to abstract stylegan / vanilla GAN, z / w latent"""
    def __init__(self, args, model, z_dim):
        super().__init__()
        self.model, self.z_dim = model, z_dim
        self.latent = args.latent
        self.is_style =  args.model.startswith('stylegan')

    def forward(self, param):
        if self.latent == 'z':
            if self.is_style:
                image = self.model(param, None)
            else:
                image = self.model(param)
        # w / wp
        else:
            assert self.is_style
            if self.latent == 'w':
                param = param.repeat(1, self.model.mapping.num_ws, 1)
            image = self.model.synthesis(param)
        return image

# 1.  NORMALIZATION MODULE
#     - This normalizes the input image so it can be put into the 
# #     Sequential function
#     - also called "min-max scaling"
#     - normalizing images puts values of pixels between range
#       of -1 and 1.
#     - Normalization does the following to each r,g,b channel:
#       image = (image - mean) / std.  This results in that
#
#     - IMPORTANCE: Normalization helps neural nets to converge,
#       also prevents mode collapse by equalizing all values (patterns
#       in image) and prevent dominance of one value over another
#       which leads to mode collapse (generator producing same thing)

# create new class which will handle the image normalization
# to pass into the Sequential function
class Normalization(nn.Module):
    # initialize the noramlization with mean & std arguments
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # define mean and std tensors
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    # create forward function to return normalized image
    # via equation previously mentioned
    def forward(self, img):
        # Normalize image with mean and std using the following
        # equation: image = (image - mean) / std
        return (image - self.mean) / self.std


# 2. PERCEPTUAL LOSS MODULE
#    - WHAT: Perceptual loss is used to determine the difference between
#      two images. 
#    - Often used to compare high level differences between images
#      such as style and content differences
#    - HOW: It sums all the squared errors between all the pixels
#      and takes the mean.    
#    - WHY: Helps to improve image quality, and when optimized
#      based on high level features extracted from the trained
#      networks, it speeds up and improves quality even more
#      resulting in higher quality images.

# create new percptual loss class
class PerceptualLoss(nn.Module):
    def __init__(self, add_layer=['conv_5']):
        super().__init__()
        # create tensor for normalization mean & stf. 
        # The rgb std / mean values below are based on ImageNet standards
        # Imagenet determined these were the best values calculated
        # from millions of images
        # send to device (GPU)
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        # create normalized image variable from the "normalization"
        # function created earlier
        # this takes in the ImageNet mean & std values above
        norm = Normalization(cnn_normalization_mean, cnn_normalization_std)
        # assign the vgg19 neural network to "cnn" variable
        # vgg19 is a pretrained network that is 19 layers deep
        # (16 conv2d layers, 3 Fully connect layer, 5 MaxPool layers and 1 SoftMax layer)
        # - it is trained on ImageNet and can classify images into 1000 categories
        cnn = vgg19(pretrained=True).features.to(device).eval()

        self.model = nn.ModuleList()
        i = 0

        #    SEQUENTIAL MODEL
        #    WHAT: A sequential module contains an ordered
        #    list of child modules such conv2d, ReLU, MaxPool2d,
        #    Conv2d, etc.  
        #    - these have to be put in the correct order
        #    WHY: As vgg19 is sequential model, a sequential 
        #    module is created so we can put in the modules
        #    (maxpool, conv2d, etc) in the right order.
        #    INPUT:  Input is "norm", the normalized image
        cur_model = nn.Sequential(norm)
        
        # Create the modules required in vgg19 and make them
        # accessible if the "isinstance" function calls them.
        # increment the i counter for every convolution layer
        # children = nn modules (conv2d, maxpool, etc.)
        for layer in cnn.children():
            # activate convolution 2d child module
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            # activate activation function child module
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            # activate maxpool2d child module
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            # activate batchNorm2d child module
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            # add in specific module that is activated
            cur_model.add_module(name, layer)
            # ** if name = conv5, set sequential (not sure I get this step...)
            if name in add_layer:
                self.model.append(cur_model)
                cur_model = nn.Sequential()

    #    FORWARD FUNCTION
    #    WHAT: The forward function takes in the predicted image
    #    and the target image and returns the loss
    def forward(self, pred, target):
        if isinstance(target, tuple):
            target, mask = target
        # reset loss as 0 for every pass
        # sets net as model to take in image to be
        #   1 - normalized
        #   2 - sent through perceptual loss
        loss = 0
        for net in self.model:
            # send in predicted image into model
            pred = net(pred)
            # send in targtet image into model
            target = net(target)
            h = pred.size(-1)
            feat_mask = F.adaptive_avg_pool2d(mask, (h,h))
            # calculate loss via MSE (between the prediction
            # image * mask and target image * mask) + reg loss
            loss = loss + F.mse_loss(pred*feat_mask, target*feat_mask)

        return loss

# 3. CRITERION CLASS
#    WHAT: A class for managing the loss criteria
class Criterion(nn.Module):
    def __init__(self, args, mask=False, layer=['conv_5']):
        super().__init__()
        self.perc_wgt = args.perc_wgt
        # add in l1 weight
        self.l1_wgt = args.l1_wgt
        self.mask = mask

        # create empty dictionary to hold all losses
        self.losses = {}
        # Set perc_wgt to perceptual loss if perc_wgt is greater than 0
        if self.perc_wgt > 0:
            self.perc = PerceptualLoss(layer)

    def forward(self, pred, target):
        """Calculate loss of prediction and target. in p-norm / perceptual  space"""
        # Part 3: Mask to Image
        if self.mask:
            target_img, mask = target
            # sets loss as loss weight(10) * pytorch L1 Loss function
            # PyTorch L1 Loss function returns the mean absolute error
            # between the prediction image * mask and target image * mask
            loss_l1 = self.l1_wgt*F.l1_loss(pred*mask, target_img*mask)
        else:
            # If not part 3 (no mask) no need to multiply by mask,thus...
            loss_l1 = self.l1_wgt*F.l1_loss(pred, target)
        # set loss_vgg to zero, unless perc_wgt > 0, as shown next
        loss_vgg = 0
        # if perc_wgt > 0, lets use Perceptual loss (improves results)
        if self.perc_wgt > 0:
            # we times the loss_vgg by perc_wgt (10) to give it more 
            # importance than discriminator loss (L1)
            loss_vgg = self.perc_wgt * self.perc(pred, target)
        # Add both the L1 discriminator loss and vgg Perceptual loss 
        # together
        # if perc_wgt = 0, it will just be the L1 loss (discriminator loss)
        # if perc_wgt > 0, it will be L1 loss + VGG loss, remember,
        # we times VGG * 10 so it will have a bigger impact on overall loss.
        loss = loss_l1 + loss_vgg
        # add current losses to losses dict
        self.losses['l1'] = loss_l1
        self.losses['vgg'] = loss_vgg
        return loss


def save_images(image, fname, col=8):
    image = image.cpu().detach()
    image = image / 2 + 0.5

    image = vutils.make_grid(image, nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        imageio.imwrite(fname + '.png', image)
    return image


def save_gifs(image_list, fname, col=1):
    """
    :param image_list: [(N, C, H, W), ] in scale [-1, 1]
    """
    image_list = [save_images(each, None, col) for each in image_list]
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    imageio.mimsave(fname + '.gif', image_list)

def sample_noise(dim, device, latent, model, N=1, from_mean=False):
    """
    sample (take the mean if from_mean=True) N noise vector (z) or N style latent(w/w+) depending on latent value.
    To generate a noise vector, just sample from a normal distribution.
    To generate a style latent, you need to map the noise (z) to the style (W) space given the `model`.
    Some hint on the z-mapping can be found at stylegan/generate_gif.py L70:81.
    :return: Tensor on device in shape of (N, dim) if latent == z
    Tensor on device in shape of (N, 1, dim) if latent == w
    Tensor on device in shape of (N, nw, dim) if latent == w+
    """

    if latent == 'z':
        vector = torch.randn([N, dim], device = device) if not from_mean else torch.zeros([N, dim], device = device)
    elif latent == 'w':
        if from_mean:
            # ** Not sure why N is set to 100000
            z = sample_noise(dim, device, 'z', None, N = 100000)
            w = model.mapping(z, None)
            # take mean of w and keep same dimensions
            w = w.mean(0, keepdim=True)
            # slice w to obtain correct value
            # per piazza post & Yufei master code
            # ** need clarification on this
            vector = w[:, 0:1]
        else:
            z = sample_noise(dim, device, 'z', None, N)
            w = model.mapping(z, None)
            # slice w to obtain correct value
            vector = w[:, 0:1]
    elif latent == 'w+':
        if from_mean:
            z = sample_noise(dim, device, 'z', None, N = 100000)
            w = model.mapping(z, None)
            # take mean of w and keep same dimensions
            w = w.mean(0, keepdim=True)
            # do not need to slice here as value is correct
            vector = w
        else:
            z = sample_noise(dim, device, 'z', None, N)
            w = model.mapping(z, None)
            # do not need to slice here as value is correct
            vector = w
    else:
        raise NotImplementedError('%s is not supported' % latent)
    return vector


def sample(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    batch_size = 16
    
    # update device depending on availability
    # added this in
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # todo: complete sample_noise and wrapper (forward pass of network)
    param = sample_noise(z_dim, device, args.latent, model, batch_size)
    image = wrapper(param)
    fname = os.path.join('output/forward/%s_%s' % (args.model, args.mode))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    save_images(image, fname)

# 4. OPTIMIZE PARAMETER
#    WHAT: This optimization function kicks off the training loop
#    - it uses all the above functions to create the following:
#           - image --> prediction image (via Wrapper function)
#           - loss -> between image (prediction) & target image (via Criterion Function)
#           - reg_loss --> 
#           - loss --> total loss (loss + regular loss)
#    OUTPUT: image & Param (Image Latent Vector - image location in latent space
#            as represented by a vector)
def optimize_para(wrapper, param, target, criterion, num_step, save_prefix=None, res=False):
    """
    wrapper: image = wrapper(z / w/ w+): an interface for a generator forward pass.
    param: z / w / w+
    target: (1, C, H, W)
    criterion: loss(pred, target)
    """
    # create tensor with zero scalars in same size as param
    # param is the random noise tensor generated by "sample_noise" 
    # function above
    delta = torch.zeros_like(param)
    delta = delta.requires_grad_().to(device)
    # LBFGS
    # set optimizer to LBFGS with "Wolfe" line search
    # WHAT: LBFGS is a way to find the local minimum of an objective 
    #       function, in this case, reducing loss between target
    #       and prediction image. 
    #       - It is kind of like gradient descent
    # HOW:  It approximates the newton updates by iteratively updating 
    #       an approximation to the inverse Hessian matrix
    optimizer = FullBatchLBFGS([delta], lr=.1, line_search='Wolfe')
    # set iteration cound to 0
    iter_count = [0]

    # create closure function
    def closure():
        # add 1 to iter_count
        iter_count[0] += 1
        # set the gradients to zero before starting to do backpropragation 
        # because PyTorch accumulates the gradients on subsequent backward 
        # passes
        optimizer.zero_grad()
        # create prediction image using wrapper class
        image = wrapper(param + delta, )
        # determine loss between prediction image & target image 
        # using criterion class
        loss = criterion(image, target)
        # define the regular loss
        reg_loss = (delta ** 2).mean()
        # add regular loss to losses dictionary
        criterion.losses['reg'] = reg_loss
        # calculate final loss (criterion loss + regular loss)
        loss = loss + reg_loss
        # save image, results and print losses after every 250th iteration of
        # optimization function loop (deterimined by # of epochs which
        # corresponds with number of batches that will be processed)
        if iter_count[0] % 250 == 0 and save_prefix is not None:
            print('iter count {} loss {:4f}'.format(iter_count, loss.item()))
            iter_result = image.data.clamp_(-1, 1)
            save_images(iter_result, save_prefix + '_%d' % iter_count[0])
            # below front master code.  
            # - prints the key in the losses dictionary
            # - will tell us whether is:
            #       - L1
            #       - vgg
            #       - reg loss
            for key in criterion.losses:
                print('%s: %f' % (key, criterion.losses[key]))
        return loss

    # call closure function accumulates gradient for each parameter
    # - parameter being the latent space of image
    # - gradient: An error gradient is the direction and magnitude 
    #   calculated during the training of a neural network that is 
    #   used to update the network weights in the right direction 
    #   and by the right amount
    loss = closure()
    loss.backward()
    # optimizer.step = performs a single optimization step or loop
    # which performs a parameter update based on current gradient 
    while iter_count[0] <= num_step:
        options = {'closure': closure, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
    image = wrapper(param)
    return param, image


# DRAW FUNCTION 
# WHAT: sketch to image function, so it uses mask, and perc_wgt
# should be set to 0
def draw(args):
    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input, args.reso, alpha=True)
    # "True" triggers mask to be used.
    criterion = Criterion(args, True)

    for idx, (rgb, mask) in enumerate(loader):
        rgb, mask = rgb.to(device), mask.to(device)
        save_images(rgb, 'output/draw/%d_data' % idx, 1)
        save_images(mask, 'output/draw/%d_mask' % idx, 1)
        # todo: optimize sketch 2 image
        # create prediciton latent vector
        param = sample_noise(z_dim, device, args.latent, model, from_mean=True)

        # Initiate training loop with optimize_para given
        # above variables (model, wrapper (prediction image), loader,
        # criterion (triggers mask), )
        optimize_para(wrapper, param, (rgb, mask), criterion, args.n_iters, 
                    'output/draw/%d_%s_%s_%g_%g' % (idx, args.model, args.latent, args.perc_wgt, args.l1_wgt))
        


def project(args):
    # load images
    loader = get_data_loader(args.input, args.reso, is_train=False)

    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    print('model {} loaded'.format(args.model))
    # todo: implement your criterion here.
    criterion = Criterion(args)
    # project each image
    for idx, (data, _) in enumerate(loader):
        target = data.to(device)
        save_images(data, 'output/project/%d_data' % idx, 1)
        param = sample_noise(z_dim, device, args.latent, model)
        optimize_para(wrapper, param, target, criterion, args.n_iters,
                      'output/project/%d_%s_%s_%g' % (idx, args.model, args.latent, args.perc_wgt))
        if idx >= 0:
            break


# INTERPOLATE
# WHAT: Slowly interpolates an image from its position in latent towards
#       the position of another image in latent space
def interpolate(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input, args.reso,)
    criterion = Criterion(args)
    # not sure what this does
    start = None
    for idx, (image, _) in enumerate(loader):

        save_images(image, 'output/interpolate/%d' % (idx))
        target = image.to(device)
        param = sample_noise(z_dim, device, args.latent, model, from_mean=True)
        param, recon = optimize_para(wrapper, param, target, criterion, args.n_iters)
        save_images(recon, 'output/interpolate/%d_%s_%s' % (idx, args.model, args.latent))

        if idx % 2 == 0:
            start = param
            continue
        last = param
        # create list of 50 evenly spaced intervals between 0 & 1
        # this will be the number of images that will be created by
        # the generator to create the gif.
        alpha_list = np.linspace(0,1,50)
        # create empty list to hold images
        image_list = []

        with torch.no_grad():
            for interp in alpha_list:
                param = interp * last + (1-interp) * last
                image = wrapper(param)
                # 4. add image to image_list until 50 images reached
                image_list.append(image.cpu())
          
        save_gifs(image_list, 'output/interpolate/%d_%s_%s' % (idx, args.model, args.latent))
        if idx >= 3:
            break
    return


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='stylegan', choices=['vanilla', 'stylegan', 'stylegan128', 'stylegan256'])
    parser.add_argument('--mode', type=str, default='sample', choices=['sample', 'project', 'draw', 'interpolate'])
    parser.add_argument('--latent', type=str, default='z', choices=['z', 'w', 'w+'])
    parser.add_argument('--n_iters', type=int, default=1000, help="number of optimization steps in the image projection")
    # add in "reso" this per Yufei master code
    parser.add_argument('--reso', type=int, default=64, help="number of optimization steps in the image projection")
    parser.add_argument('--perc_wgt', type=float, default=0.01, help="perc loss lambda")
    parser.add_argument('--l1_wgt', type=float, default=10., help="perc loss lambda")
    parser.add_argument('--input', type=str, default='data/cat/*.png', help="path to the input image")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    if args.mode == 'sample':
        sample(args)
    elif args.mode == 'project':
        project(args)
    elif args.mode == 'draw':
        draw(args)
    elif args.mode == 'interpolate':
        interpolate(args)













