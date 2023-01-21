
# USAGE - MINE
# python3 1_image_processing.py --image data/1_raw/1_soho_raw
# --output 2_data/2_processed/1_soho_processed --total 2


# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from PIL import Image
import numpy as np
import argparse
import glob
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store augmentation examples")
ap.add_argument("-t", "--total", type=int, default=10,
	help="# of training samples to generate")
args = vars(ap.parse_args())

print(args["image"])

# for filename in os.listdir(args["image"]):
#     if filename.endswith(".jpg"): 

# create loop for each file in unlabeled_data_path that is jpg
for imagePath in glob.glob(f'{args["image"]}/*.jpg'):

    # load the input image, convert it to a NumPy array, and then
    # reshape it to have an extra dimension
    print("[INFO] loading example image...")
    image = load_img(imagePath, target_size = (128, 128), interpolation="bicubic")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # construct the image generator for data augmentation then
    # initialize the total number of images generated thus far
    aug = ImageDataGenerator(
        #rotation_range=30,
        zoom_range=[0.85, 1.0],
        #width_shift_range=0.075,
        #height_shift_range=0.075,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

    total = 0

    # construct the actual Python generator
    print("[INFO] generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
        save_prefix="image", save_format="png")

    # loop over examples from our image data augmentation generator
    for image in imageGen:
        # increment our counter
        total += 1

        # if we have reached the specified number of examples, break
        # from the loop
        if total == args["total"]:
            break