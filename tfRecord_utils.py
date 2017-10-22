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
import tensorflow as tf

# image supposed to have shape: 480 x 640 x 3 = 921600
IMAGE_PATH = 'data_val'
TFR_PATH = 'data_TFrecord'
TFR_FILE = 'val_TFrecord.tf'
IMG_W = 150
IMG_H = 60
char_set = ('_0123456789'
            'abcdefghijklmnopqrstuvwxyz'
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
char_map = {char:idx for idx, char in enumerate(char_set)}
#char_map['_'] = 0 #Padding character

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_image_binary(filename):
    """ You can read in the image using tensorflow too, but it's a drag
        since you have to create graphs. It's much easier using Pillow and NumPy
    """
    image = (Image.open(filename)
                .resize((IMG_W, IMG_H), resample=Image.BILINEAR))
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    #plt.imshow(image)
    #plt.show()
    #print(shape)
    return shape.tobytes(), image.tobytes() # convert image to raw data bytes in the array.

#def write_to_tfrecord(label, shape, binary_image, tfrecord_file):
#    """ This example is to write a sample to TFRecord file. If you want to write
#    more samples, just use a loop.
#    """
#    writer = tf.python_io.TFRecordWriter(tfrecord_file)
#    # write label, shape, and image content to the TFRecord file
#    example = tf.train.Example(features=tf.train.Features(feature={
#                'label': _int64_feature(label),
#                'shape': _bytes_feature(shape),
#                'image': _bytes_feature(binary_image)
#                }))
#    writer.write(example.SerializeToString())
#    writer.close()

def write_to_tfrecord(label, label_string, length, binary_image, writer):
    """ This example is to write a sample to TFRecord file. If you want to write
    more samples, just use a loop.
    """
    # write label, shape, and image content to the TFRecord file
    example = tf.train.Example(features=tf.train.Features(feature={
                'label': _bytes_feature(label),
                #'label_string': _bytes_feature(label_string),
                #'length': _int64_feature(length),
                #'shape': _bytes_feature(shape),
                'image': _bytes_feature(binary_image)
                }))
    writer.write(example.SerializeToString())

#def write_tfrecord(label, image_file, tfrecord_file):
#    shape, binary_image = get_image_binary(image_file)
#    write_to_tfrecord(label, shape, binary_image, tfrecord_file)

def read_file_format(tfrecord_file_queue):
    """Read and decode file format here
    Input: string_input_produce from tf
    Output: Tensor for example"""
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as 
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={                                
                            'label': tf.FixedLenFeature([], tf.string),
                            #'label_string': tf.FixedLenFeature([], tf.string),
                            #'length': tf.FixedLenFeature([], tf.int64),
                            #'shape': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string),
                        }, name='features')
    # image was saved as uint8, so we have to decode as uint8.
    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    label = tf.decode_raw(tfrecord_features['label'], tf.int32)
    #label_string = tfrecord_features['label_string']
    #length = tfrecord_features['length']
    
    #shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    # the image tensor is flattened out, so we have to reconstruct the shape
    image = tf.reshape(image, (IMG_H, IMG_W, 3))
    label = tf.reshape(label, (7,)) #To force size specification
    return image, label
    
def write_tfrecord(label, label_string, length, image_file, writer):
    shape, binary_image = get_image_binary(image_file)
    write_to_tfrecord(label, label_string, length, binary_image, writer)

def read_from_tfrecord(filenames, batch_size=None, shuffle=True):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, 
                                                         name='queue',
                                                         shuffle=shuffle)
    #label = tfrecord_features['label']
    image, label = read_file_format(tfrecord_file_queue)
    if batch_size: #Return multiple samples
        image, label = read_file_format(tfrecord_file_queue)
        #example_list = [read_file_format(fil)]
    
        # See recommendations in http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf
        min_after_dequeue = 10 * batch_size
        capacity = 20 * batch_size 
        images, labels = tf.train.shuffle_batch([image, label], 
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                min_after_dequeue=min_after_dequeue,)
                                                #num_threads=4)
        return images, labels
    else:
        image, label = read_file_format(tfrecord_file_queue)
        return image, label #, label_string, length

def read_batch_tfrecords(tfrecord_files, batch_size=32):
    images, labels = read_from_tfrecord(tfrecord_files, batch_size=32)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for j in range(5):
            images_np, labels_np = sess.run([images, labels])
            #FOR DEBUGGING
            for i in range(batch_size):
                print(labels_np[i])
                print(''.join(char_set[idx] for idx in labels_np[i]).rstrip('_'))
                plt.imshow(images_np[i])
                plt.show()
        coord.request_stop()
        coord.join(threads)
    

    
    
def read_tfrecord(tfrecord_file):
    image, label = read_from_tfrecord([tfrecord_file])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        image, label = sess.run([image, label])
        #Reverse the label to string
        label_string = ''.join(char_set[idx] for idx in label).rstrip('_')
        length = len(label_string)
        #label_string = label_string.decode('utf-8')
        coord.request_stop()
        coord.join(threads)
    print(label)
    print(label_string)
    print(length)
    plt.imshow(image)
    plt.show() 

def main():
    #Grab files here
    image_files = list(os.walk(IMAGE_PATH))[1:]
    
    TFR_FILEPATH = os.path.join(TFR_PATH, TFR_FILE)
    writer = tf.python_io.TFRecordWriter(TFR_FILEPATH)
    for path, _, files in tqdm(image_files, desc='Folder loop'):
        #folder_number = os.path.split(path)[-1]
        #TFR_FILE_PATH = os.path.join(TFR_PATH, TFR_FILE + folder_number)
        #writer = tf.python_io.TFRecordWriter(TFR_FILE_PATH)
        for f in tqdm(files, desc='File loop', leave=False):
            #Insert transformation here - Convert label to DATA
            label_string = f[:-4]
            #label = 1
            length = len(label_string)
            label = np.array([char_map[char] for char in label_string.ljust(7, '_')]).astype(np.int32)
            write_tfrecord(label.tobytes(),
                           label_string.encode('utf-8'),
                           length,
                           os.path.join(path, f),
                           writer)
            #break
        #break
        #writer.close()
    writer.close()
   

if __name__ == '__main__':
    
    #main() 
        
    #Now try reading in batches
    file_names = [os.path.join(TFR_PATH, f) for f in os.listdir(TFR_PATH) if f.startswith('train')]
    file_names = ['data_TFrecord/val_TFrecord.tf']
    #Batch reading - TEST
    read_batch_tfrecords(file_names, batch_size=16)
    
