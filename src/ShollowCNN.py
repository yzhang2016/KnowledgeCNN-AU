#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:32:05 2017

@author: yong
"""



import tensorflow as tf 
import numpy as np
from functools import reduce 
import bp4d_input
import ShollowCNN_AUModel


IMG_SIZE = bp4d_input.IMAGE_SIZE
NUM_CLASS = bp4d_input.NUM_CLASSES

W_DECAY = 1e-7 /ShollowCNN_AUModel.BATCH_SIZE   # fully connected layer

class ShollowCNN:
    """
    A trainable version of shollow CNN with three convolutional layers
    """
    def __init__(self, model_npy_path=None,dropout=1.0):
        if model_npy_path is not None:
            self.data_dict = np.load(model_npy_path,encoding='latin1').item()
            
        else:
            self.data_dict = None
            
        self.var_dict = {}
        self.dropout = dropout
        
    def inference(self,rgb,train_mode=None,is_build=True):
        """
        Load variable from npy to build the AlexNet
        rgb: [batch,height,width,3]
        train_model: if true, dropout will be used
        """  
        red, green, blue = tf.split(value=rgb,num_or_size_splits=3,axis=3)
        assert red.get_shape().as_list()[1:] == [IMG_SIZE,IMG_SIZE,1]
        assert green.get_shape().as_list()[1:] == [IMG_SIZE,IMG_SIZE,1]
        assert blue.get_shape().as_list()[1:] == [IMG_SIZE,IMG_SIZE,1]
        
        bgr = tf.concat(axis=3,values=[blue,green,red])
        assert bgr.get_shape().as_list()[1:] == [IMG_SIZE,IMG_SIZE,3]


        self.conv1 = self.conv_layer_relu(bgr,5,1,3,32,'conv1',
                                          w_std=0.001, b_mean=0.0,
                                          trainable=True,is_build=is_build)
        self.pool1 = self.max_pool(self.conv1,3,2,'pool1')    
        
        self.conv2 = self.conv_layer_relu(self.pool1,5,1,32,64,'conv2',
                                          w_std=0.001, b_mean=0.0,
                                          trainable=True,is_build=is_build)
        self.pool2 = self.max_pool(self.conv2,3,2,'pool2')
        
        self.conv3 = self.conv_layer_relu(self.pool2,5,1,64,128,'conv3',
                                          w_std=0.001, b_mean=0.0,
                                          trainable=True,is_build=is_build)
        self.pool3 = self.max_pool(self.conv3,3,2,'pool3')
        
        self.fc4 = self.fc_layer_relu(self.pool3, 8192, 128,'fc4',
                                      w_std=0.001, b_mean=0.1,
                                      trainable=True,w_decay=W_DECAY,is_build=is_build)
        
        if train_mode is not None:
            self.fc4 = tf.cond(train_mode,lambda:tf.nn.dropout(self.fc4,self.dropout), lambda:self.fc4)
        else:
            self.fc4 = tf.nn.dropout(self.fc4,self.dropout)
                
        self.fc5 = self.fc_layer(self.fc4,128,NUM_CLASS,name='fc5',
                                 w_std=0.001, b_mean=0.0,
                                 trainable=True,w_decay=0.0,is_build=is_build)
        
        self.data_dict = None
        if is_build:
            self.activation_summary(self.conv1)  
            self.activation_summary(self.conv2)
            self.activation_summary(self.conv3)
            self.activation_summary(self.fc4)
        
        return self.fc5, self.fc4

    def max_pool(self,bottom,filter_size,stride_size,name):
        return tf.nn.max_pool(bottom,ksize=[1,filter_size,filter_size,1],
                              strides=[1,stride_size,stride_size,1],
                              padding='SAME',name=name)
        
    def avg_pool(self,bottom,filter_size,stride_size,name):
        return tf.nn.avg_pool(bottom,ksize=[1,filter_size,filter_size,1],
                              strides=[1,stride_size,stride_size,1],
                              padding='SAME',name=name)

    def conv_layer_relu(self,bottom,filter_size,stride_size,
                        in_chs,out_chs,name,
                        w_std=0.001, b_mean=0.1,
                        trainable=True,w_decay=None,is_build=True):
        with tf.variable_scope(name):
            filters,biases = self.get_conv_var(filter_size,
                                               in_chs,out_chs,name,
                                               w_std, b_mean,
                                               trainable,w_decay,is_build)
            
            conv = tf.nn.conv2d(bottom,filters,[1,stride_size,stride_size,1],padding='SAME')
            bias = tf.nn.bias_add(conv,biases)
            relu = tf.nn.relu(bias)
            
            return relu
        
    def conv_layer_relu_split(self,bottom,filter_size,stride_size,
                              in_chs,out_chs,name,
                              w_std=0.001, b_mean=0.1,
                              trainable=True,w_decay=None,is_build=True):
        """ Split the input and the kernel into two parts and convoluate sperately 
        This way can get less parameters by slicing the thick input into two thin inputs
        """
        with tf.variable_scope(name):
            filters,biases = self.get_conv_var(filter_size,
                                               in_chs,out_chs,name,
                                               w_std, b_mean,
                                               trainable,w_decay,is_build)
            
            bottom_group = tf.split(bottom,2,axis=3)
            filter_group = tf.split(filters,2,axis=3)
            
            output_group = [tf.nn.conv2d(b,f,[1,stride_size,stride_size,1],padding='SAME') 
                            for (b,f) in zip(bottom_group,filter_group)]
            conv = tf.concat(output_group,axis=3)
            bias = tf.nn.bias_add(conv,biases)
            relu = tf.nn.relu(bias)
            
            return relu

      
    def conv_layer_relu_lrn(self,bottom,filter_size,stride_size,
                            in_chs,out_chs,name,
                            w_std=0.001, b_mean=0.1,
                            trainable=True,w_decay=None,is_build=True):
        with tf.variable_scope(name):
            filters,biases = self.get_conv_var(filter_size,
                                               in_chs,out_chs,name,
                                               w_std, b_mean,
                                               trainable,w_decay,is_build)
            
            conv = tf.nn.conv2d(bottom,filters,[1,stride_size,stride_size,1],padding='SAME')
            bias = tf.nn.bias_add(conv,biases)
            relu = tf.nn.relu(bias)
            radius  = 2 
            alpha = 2e-05
            beta = .75
            bias = 1.
            lrn = tf.nn.lrn(relu, depth_radius=radius,alpha=alpha,beta=beta,bias=bias)
            
            return lrn
        
    def conv_layer_relu_lrn_split(self,bottom,filter_size,stride_size,
                                  in_chs,out_chs,name,
                                  w_std=0.001, b_mean=0.1,
                                  trainable=True,w_decay=None,is_build=True):
        """ Split the input and the kernel into two parts and convoluate sperately 
        This way can get less parameters by slicing the thick input into two thin inputs
        """
        with tf.variable_scope(name):
            filters,biases = self.get_conv_var(filter_size,
                                               in_chs,out_chs,name,
                                               w_std, b_mean,
                                               trainable,w_decay,is_build)
            
            bottom_group = tf.split(bottom,2,axis=3)
            filter_group = tf.split(filters,2,axis=3)
            
            output_group = [tf.nn.conv2d(b,f,[1,stride_size,stride_size,1],padding='SAME') 
                            for (b,f) in zip(bottom_group,filter_group)]
            conv = tf.concat(output_group,axis=3)
            bias = tf.nn.bias_add(conv,biases)
            relu = tf.nn.relu(bias)
            
            radius  = 2 
            alpha = 2e-05
            beta = .75
            bias = 1.
            lrn = tf.nn.lrn(relu, depth_radius=radius,alpha=alpha,beta=beta,bias=bias)
            
            return lrn 
        
    def fc_layer(self,bottom,in_size,out_size,name,
                 w_std=0.001, b_mean=0.1,
                 trainable=True,w_decay=None,is_build=True):
        with tf.variable_scope(name):
            weights,biases = self.get_fc_var(in_size,out_size,name,
                                             w_std, b_mean,
                                             trainable,w_decay,is_build)
            
            x = tf.reshape(bottom,[-1,in_size])
            fc = tf.nn.bias_add(tf.matmul(x,weights),biases)
            
            return fc

    def fc_layer_relu(self,bottom,in_size,out_size,name,
                      w_std=0.001, b_mean=0.1,
                      trainable=True,w_decay=None,is_build=True):
        with tf.variable_scope(name):
            weights,biases = self.get_fc_var(in_size,out_size,name,
                                             w_std, b_mean,
                                             trainable,w_decay,is_build)
            
            x = tf.reshape(bottom,[-1,in_size])
            fc = tf.nn.bias_add(tf.matmul(x,weights),biases)
            relu = tf.nn.relu(fc)
            
            return relu
        
    def get_conv_var(self,filter_size,in_chs,out_chs,name,
                     w_std, b_mean,
                     trainable,w_decay,is_build):
        initial_value = tf.truncated_normal([filter_size,filter_size,in_chs,out_chs],mean=.0,stddev=w_std)
        filters = self.get_var(initial_value,name,0, name+'_filters',trainable,w_decay,is_build)
        
        initial_value = tf.truncated_normal([out_chs],mean=b_mean,stddev=.0)
        biases = self.get_var(initial_value,name,1,name+'_biases',trainable,w_decay=None,is_build=is_build)
        
        return filters, biases
        
    def get_fc_var(self,in_size,out_size,name,
                   w_std, b_mean,
                   trainable,w_decay,is_build):
        initial_value = tf.truncated_normal([in_size,out_size],mean=.0,stddev=w_std)
        weights = self.get_var(initial_value,name,0,name+'_weights',trainable,w_decay,is_build)
        
        initial_value = tf.truncated_normal([out_size],mean=b_mean,stddev=.0)
        biases = self.get_var(initial_value,name,1,name+'_biases',trainable,w_decay=None,is_build=is_build)
        
        return weights, biases
    
    def get_var(self,initial_value,name,idx,var_name,trainable,w_decay,is_build):
        if is_build:
            if self.data_dict is not None and name in self.data_dict:
                value = self.data_dict[name][idx]
            else:
                value = initial_value
                
            if trainable:
                var = tf.Variable(value,name=var_name)
                if w_decay is not None:
                    weight_decay = tf.multiply(tf.nn.l2_loss(var),w_decay,name='weight_loss')
                    tf.add_to_collection('losses',weight_decay)
            else:
                var = tf.constant(value,dtype=tf.float32,name=var_name)
                
            self.var_dict[(name,idx)] = var
            return var
        else:
            return self.var_dict[(name,idx)]
        
    def save_npy(self,sess,npy_path='./AlexNet-save.npy'):
        assert isinstance(sess,tf.Session)
        data_dict = {}
        
        for (name,idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out
        np.save(npy_path,data_dict)
        print(('file saved:'), npy_path)
        return npy_path
    
    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count+= reduce(lambda x,y: x*y, v.get_shape().as_list())
        return count
    
    def activation_summary(self,x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name+'/activations',x)
        tf.summary.scalar(tensor_name+'/sparsity',tf.nn.zero_fraction(x))
    