#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:59:20 2017

@author: yong
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 09:40:15 2017

@author: yong
"""

import os 
import glob
import tensorflow as tf
import numpy as np 
#import variables
rng = np.random

AU = [6,10,12,14,17] # for the filename
#
#AUIND = variables.AUIND
#AUNAME = AU[AUIND]

IMAGE_SIZE = 32
NUM_CLASSES = 5
NUM_EXAMPLE_PER_EPOCH_FOR_TRAIN = 750000 * 5 
NUM_EXAMPLE_PER_EPOCH_FOR_EVAL = 75000

# FERA : training and testing
SUB_F_train = np.arange(1,24,2).tolist()
SUB_F_test = np.arange(2,24,2).tolist()
SUB_M_train = np.arange(1,19,2).tolist()
SUB_M_test = np.arange(2,19,2).tolist()

#
#SUB_F_test = SUB_F_train
#SUB_M_test = SUB_M_train

SUB_F_train = SUB_F_train[0:7]
SUB_M_train = SUB_M_train[0:6]


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('imgpath','../../data/color_64/',
                           """The image path""")

def gen_target_height(img_size):
    
    
    rnd_val0 = tf.random_uniform([],minval=0., maxval=1.)
    
    rnd_val = tf.random_uniform([],minval=0.95, maxval=1.05)
    
    A = tf.scalar_mul(rnd_val,tf.constant(img_size,tf.float32))
    B = tf.constant(img_size,tf.float32)
    
    t_height = tf.cond(rnd_val0 > 0.5 , lambda:A, lambda: B)
    
    return tf.cast(t_height,tf.int32)

def read_bp4d_train(filename_queue):
    """ 1. Read filename from filename_queue, 
        2. Read image and label
    """
    class BP4DRecord(object): # like a struct
        pass 
    res = BP4DRecord()
    name_bytes = 12 * 5
    int_bytes = 6
#    task_bytes = 1
    

    num_bytes = name_bytes + int_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=num_bytes)
    key, val = reader.read(filename_queue)
    
    name = [tf.substr(val,12*i,12)  for i in range(0,5)]
    
    names = [tf.string_join([i,'.jpg']) for i in name]
    
#    print(names)
    
    filepaths = [tf.string_join([FLAGS.imgpath,i],separator='/') for i in names]
    images = [tf.image.decode_jpeg(tf.read_file(i),channels=3) for i in filepaths] 
    rec_bytes = tf.decode_raw(val,tf.uint8)
    ints =  [tf.cast(tf.strided_slice(rec_bytes,[name_bytes+i],[name_bytes+i+1]),tf.int32) for i in range(int_bytes)]
    
    # get the subject 
    SUB = tf.string_to_number(tf.substr(name[0],1,3),out_type=tf.int32) 
    
    
    
    '''====> attention to change <==='''
    res.ILN_dict = {'IS': images[0], 
                    'IA': images[1],
                    'IB': images[2],
                    'IE': images[3],  
                    'IN': images[4],
                    'LS': ints[0], 
                    'LA': ints[1] ,
                    'LB': ints[2] ,
                    'LE': ints[3],  
                    'LN': ints[4],
                    'NS': name[0], 
                    'NA': name[1],
                    'NB': name[2],
                    'NE': name[3],  
                    'NN': name[4],
                    'T' : ints[5],
                    'SUB':SUB}
    
    res.height = IMAGE_SIZE
    res.width = IMAGE_SIZE
    res.depth = 3
    
    return res
    
    
def _generate_image_and_label_batch_train(indict,min_queue_examples,batch_size,shuffle):
    
    if shuffle:
        num_preprocess_threads = 2
        outdict = tf.train.shuffle_batch(indict,
                                       batch_size=batch_size,
                                       num_threads=num_preprocess_threads,
                                       capacity=min_queue_examples+3*batch_size,
                                       min_after_dequeue=min_queue_examples)
    else:
        num_preprocess_threads = 1
        outdict = tf.train.batch(indict,
                                       batch_size=batch_size,
                                       num_threads=num_preprocess_threads,
                                       capacity=min_queue_examples+3*batch_size)
        
    tf.summary.image('imageS',outdict['IS'])
    tf.summary.image('imageA',outdict['IA'])
    tf.summary.image('imageB',outdict['IB'])
    tf.summary.image('imageE',outdict['IE'])
    tf.summary.image('imageN',outdict['IN'])
    
    return outdict


def read_bp4d_eval(filename_queue):
    """ 1. Read filename from filename_queue, 
        2. Read image and label
    """
    class BP4DRecord(object): # like a struct
        pass 
    res = BP4DRecord()
    name_bytes = 12
    occ_bytes = 12
    int_bytes = 5
    

    num_bytes = name_bytes + occ_bytes + int_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=num_bytes)
    key, val = reader.read(filename_queue)
    
    name = tf.substr(val,0,name_bytes)
    res.name = name
    name = tf.string_join([name,'.jpg'])
    filepath = tf.string_join([FLAGS.imgpath,name],separator='/')
    res.image = tf.image.decode_jpeg(tf.read_file(filepath),channels=3)  

    rec_bytes = tf.decode_raw(val,tf.uint8)
    res.occ =  [tf.cast(tf.strided_slice(rec_bytes,[name_bytes+i],[name_bytes+i+1]),tf.int32) for i in range(occ_bytes)]
    res.occ = tf.concat([tf.reshape(i,[1]) for i in res.occ],axis=0)
    res.int = [tf.cast(tf.strided_slice(rec_bytes,[name_bytes+occ_bytes+i],[name_bytes+occ_bytes+i+1]),tf.int32) for i in range(int_bytes)]
    res.int = tf.concat([tf.reshape(i,[1]) for i in res.int],axis=0)

    res.height = 64
    res.width = 64
    res.depth = 3
    
    return res

def _generate_image_and_label_batch_eval(image,label,name,min_queue_examples,batch_size,shuffle):
    
    if shuffle:
        num_preprocess_threads = 2
        images,labels,names = tf.train.shuffle_batch([image,label,name],
                                               batch_size=batch_size,
                                               num_threads=num_preprocess_threads,
                                               capacity=min_queue_examples+3*batch_size,
                                               min_after_dequeue=min_queue_examples)
    else:
        num_preprocess_threads = 1
        images,labels,names = tf.train.batch([image,label,name],
                                       batch_size=batch_size,
                                       num_threads=num_preprocess_threads,
                                       capacity=min_queue_examples+3*batch_size)
        
    tf.summary.image('images',images)
    
    return images,labels,names

def img_process(img):
    # image process
    res_img = tf.cast(img,tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    res_img = tf.image.resize_images(res_img,[height,width])
    
    # warping
    t_height = gen_target_height(height)
    res_img = tf.image.resize_image_with_crop_or_pad(res_img,t_height,t_height)
    res_img = tf.image.resize_images(res_img,[height,width])
    
    # randomly flip 
    res_img = tf.image.random_flip_left_right(res_img) 
    res_img = tf.image.per_image_standardization(res_img)
    
    res_img.set_shape([height,width,3])
    
    return res_img

def img_process_org(img):
    # image process
    res_img = tf.cast(img,tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    res_img = tf.image.resize_images(res_img,[height,width])
    res_img = tf.image.per_image_standardization(res_img)
    
    res_img.set_shape([height,width,3])
    
    return res_img



def distorted_inputs(data_dir,batch_size):
    
    filenamesF = []
    filenamesM = []
    for j in AU: 
        for i in SUB_F_train:
            nms = glob.glob(os.path.join(data_dir,'AU%d/BP4D_tuple_F%03d*' %(j,i)))  
#            print(nms)
            filenamesF = filenamesF + nms
        for i in SUB_M_train:
            nms = glob.glob(os.path.join(data_dir,'AU%d/BP4D_tuple_M%03d*' %(j,i)))  
            filenamesM = filenamesM + nms 

    filenames = filenamesF + filenamesM
#    
#    print(filenames)
    
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' + f)
            
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    
    res = read_bp4d_train(filename_queue)
    
    # image process
    IS = img_process(res.ILN_dict['IS'])
    IA = img_process(res.ILN_dict['IA'])
    IB = img_process(res.ILN_dict['IB'])
    IE = img_process(res.ILN_dict['IE'])
    IN = img_process(res.ILN_dict['IN'])
    
    
    # fliped images
    FIS = tf.image.random_flip_left_right(IS) 
    FIA = tf.image.random_flip_left_right(IA)
    FIB = tf.image.random_flip_left_right(IB)
    FIE = tf.image.random_flip_left_right(IE)
    FIN = tf.image.random_flip_left_right(IN)
    
    
    # process labels 
    LS = tf.reshape(res.ILN_dict['LS'],[1])
    LA = tf.reshape(res.ILN_dict['LA'],[1])
    LB = tf.reshape(res.ILN_dict['LB'],[1])
    LE = tf.reshape(res.ILN_dict['LE'],[1])
    LN = tf.reshape(res.ILN_dict['LN'],[1])

    # process names
    NS = tf.reshape(res.ILN_dict['NS'],[1])
    NA = tf.reshape(res.ILN_dict['NA'],[1])
    NB = tf.reshape(res.ILN_dict['NB'],[1])
    NE = tf.reshape(res.ILN_dict['NE'],[1])
    NN = tf.reshape(res.ILN_dict['NN'],[1])    
    
    TT = tf.reshape(res.ILN_dict['T'],[1])
    SUB = tf.reshape(res.ILN_dict['SUB'],[1])
    new_dict = {'IS':IS,   'IA':IA,   'IB':IB,   'IE':IE,   'IN':IN, 
                'FIS':FIS, 'FIA':FIA, 'FIB':FIB, 'FIE':FIE, 'FIN':FIN, 
                'LS':LS,   'LA':LA,   'LB':LB,   'LE':LE, 'LN':LN, 
                'NS':NS,   'NA':NA,   'NB':NB,   'NE':NE, 'NN':NN,
                'T' :TT,   'SUB':SUB}
    
    # ensure that the random shuffling has good mixing properties
    min_fraction_of_examples_in_queue = 0.004
    min_queue_examples = int(NUM_EXAMPLE_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    
    print('Filling queue with %d BP4D images before starting to train.'
          'This will take a few minutes.' % min_queue_examples)

    outdict =  _generate_image_and_label_batch_train(new_dict, min_queue_examples,batch_size,shuffle=True)
    imgs =     {'S':outdict['IS'], 'A':outdict['IA'], 'B':outdict['IB'], 'E':outdict['IE'], 'N':outdict['IN']}
    flipimgs = {'S':outdict['FIS'], 'A':outdict['FIA'], 'B':outdict['FIB'], 'E':outdict['FIE'], 'N':outdict['FIN']}
    lbs =      {'S':outdict['LS'], 'A':outdict['LA'], 'B':outdict['LB'], 'E':outdict['LE'], 'N':outdict['LN']}
    nms =      {'S':outdict['NS'], 'A':outdict['NA'], 'B':outdict['NB'], 'E':outdict['NE'], 'N':outdict['NN']}
    tsk =      {'T':outdict['T']}
    sub =      {'SUB':outdict['SUB']}
    return imgs, flipimgs, lbs, nms, tsk,sub

def inputs_eval(data_dir,batch_size,evalShuffle):
    """ Not useful for AU intensity estiamtion: 
    """
    
    filenamesF = [os.path.join(data_dir,'BP4D_single_F%03d.bin' %i) for i in SUB_F_test]
    filenamesM = [os.path.join(data_dir,'BP4D_single_M%03d.bin' %i) for i in SUB_M_test]
    filenames = filenamesF + filenamesM
    num_examples_per_epoch = NUM_EXAMPLE_PER_EPOCH_FOR_EVAL
        
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' +f)
    
    # create queue

    filename_queue = tf.train.string_input_producer(filenames,shuffle=evalShuffle)
    
    res = read_bp4d_eval(filename_queue)
    
    res.image = img_process_org(res.image)
    
    # ensure that the random shuffling has good mixing properties
    min_fraction_of_examples_in_queue = 0.005
    min_queue_examples = int( num_examples_per_epoch * min_fraction_of_examples_in_queue)   
 
    return _generate_image_and_label_batch_eval(res.image,res.int, res.name,
                                           min_queue_examples,batch_size,
                                           shuffle=evalShuffle)
