#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 12:47:44 2017

@author: yong
"""


import os
import matplotlib.pyplot as plt
import tensorflow as tf 
import bp4d_input_weak as bp4d_input
import scipy.io as scipIO

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir','../../data/tupleFileNames_MT/',
                           """Data path""")
tf.app.flags.DEFINE_integer('batch_size',64,
                           """Data path""")

with tf.Graph().as_default():

    imgs,flipimgs,lbs,nms,tsk, sub = bp4d_input.distorted_inputs(FLAGS.data_dir,FLAGS.batch_size)
#    images,ints,names = bp4d_input.distorted_inputs(FLAGS.data_dir,FLAGS.batch_size)
#    images,ints,names = bp4d_input.inputs_eval(FLAGS.data_dir,FLAGS.batch_size,evalShuffle=False)
    
    SUB = tf.cast(sub['SUB'],tf.float32)
    A = tf.sign(12.0-SUB) + 1
    s_oneK = A / 2  # sign: choosen 1, not chosen : 0
    sum_s_oneK = tf.reduce_sum(s_oneK)

    

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        o_imgs,o_flipimgs,o_lbs,o_nms,o_tsk,o_sub, s_one , sum_s_one= sess.run([imgs,flipimgs,lbs,nms,tsk,sub,s_oneK,sum_s_oneK])
        

        scipIO.savemat('./inputName.mat',o_nms)
        scipIO.savemat('./inputLbs.mat',o_lbs)
        coord.request_stop()
        coord.join(threads)
        
        
        
        
        