""" Examples to demonstrate how to write an image file to a TFRecord,
and how to read a TFRecord file using TFRecordReader.
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
sys.path.append('..')

from tqdm import tqdm #Progress bar

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# image supposed to have shape: 480 x 640 x 3 = 921600
IMAGE_PATH = 'data'
OUTPUT_PATH = 'data_npy'
IMG_W = 150
IMG_H = 60
char_set = ('_0123456789'
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
char_map = {char:idx for idx, char in enumerate(char_set)}
#char_map['_'] = 0 #Padding character

def get_image(filename):
    """ You can read in the image using tensorflow too, but it's a drag
        since you have to create graphs. It's much easier using Pillow and NumPy
    """
    image = (Image.open(filename)
                .resize((IMG_W, IMG_H), resample=Image.BILINEAR))
    image = np.asarray(image, np.uint8)
    #shape = np.array(image.shape, np.int32)
    #plt.imshow(image)
    #plt.show()
    #print(shape)
    return image # convert image to raw data bytes in the array.

def main():
    #Grab files here

    #return labels
            #break
        #break
    print()
   

if __name__ == '__main__':
    
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    
    image_files = list(os.walk(IMAGE_PATH))[1:]
        
    labels = []
    full_files = []
    for path, _, files in image_files:
        for f in files:
            #Insert transformation here - Convert label to DATA
            label_string = f[:-4]
            #length = len(label_string)
            label = np.array([char_map[char] for char in label_string.ljust(7, '_')]).astype(np.int32)
            labels.append(label)
            full_files.append(os.path.join(path, f))
    
    labels = np.array(labels)
    images = np.array([get_image(f) for f in tqdm(full_files)])
    
    np.save(os.path.join(OUTPUT_PATH, 'train_y.npy'), labels)
    np.save(os.path.join(OUTPUT_PATH, 'train_X.npy'), images)
    
    #Batch reading - TEST

    
