#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed May  3 16:46:41 2017

@author: yong
"""

import tensorflow as tf 
import numpy as np 
import bp4d_input_weak as bp4d_input
import variables as vars

RC = vars.RC # annotation rate control 



IMAGE_SIZE = bp4d_input.IMAGE_SIZE
NUM_CLASSES = bp4d_input.NUM_CLASSES
NUM_EXAPLES_PER_EPOCH_FOR_TRAIN = bp4d_input.NUM_EXAMPLE_PER_EPOCH_FOR_TRAIN
NUM_EXAPLES_PER_EPOCH_FOR_EVAL = bp4d_input.NUM_EXAMPLE_PER_EPOCH_FOR_EVAL

BATCH_SIZE = 64
#BATCH_SIZE = 200
#
# parameters for traininig process
MOVING_AVERAGE_DECAY = 0.99999
NUM_EPOCHS_PER_DECAY = 2.0 * 0.2
LEARNING_RATE_DECAY_FACTOR = 0.1
#INITIAL_LEARNING_RATE = 0.0005
INITIAL_LEARNING_RATE = 0.00005

POS_MARGIN_FC_RANK = 0.01 
NEG_MARGIN_FC_RANK = 0.0
POS_MARGIN_LB_RANK = 0.01
NEG_MARGIN_LB_RANK = 0.0
POS_MARGIN_NEU = 0.1
NEG_MARGIN_NEU = 0.0


W_LBADD = 0.2

#W_LB = 1.0 
#W_LB_RANK = 0.5
#W_FC_RANK = 1e-2
#W_SYM = 1e-1 
#W_NEU = 1e-1


W_LB = 1.0 
W_LB_RANK = 0.2
W_FC_RANK = 1e-2 *2
W_SYM = 1e-1
W_NEU = 1e-0


W_LB_ROUND = 1.0


def comp_loss(fcs,fcs_flip,logits,labels):
    """ Compute all the loss
    """
    lb_loss(logits,labels)
#    lb_loss2(logits,labels)

#    lb_round_loss(logits,labels)
    
    triplet_lb_loss(logits,labels)
    triplet_fc_loss(fcs,labels)
    sym_loss(fcs,fcs_flip)
    neu_loss(fcs,labels)
    
    
    print('success')
    
    return tf.add_n(tf.get_collection('losses'),name='total_loss')
    
def loss_eval(logits,labels):
    """compute MAE and MSE
    """
    labels = tf.cast(labels,tf.float32)
#    labels = tf.strided_slice(labels,[0,AUIND],[BATCH_SIZE,AUIND+1])
    
    MSE = tf.reduce_mean(tf.reduce_sum(tf.square(labels-logits),1))
    MAE = tf.reduce_mean(tf.reduce_sum(tf.abs(labels-logits),1))
    return MSE,MAE

def triplet_lb_loss(logits,labels):
    ma = tf.cast(tf.equal(labels['S'], labels['E']),tf.float32) * NEG_MARGIN_LB_RANK
    mb = tf.cast(tf.not_equal(labels['S'], labels['E']),tf.float32) * POS_MARGIN_LB_RANK   
    margin =  ma + mb

    # S < A < B < E ; N 
    d1_pos = tf.square(logits['S']-logits['A'])
    d1_neg = tf.square(logits['S']-logits['B'])
    loss1 = tf.maximum(0., margin + d1_pos - d1_neg)
    loss1 = tf.reduce_mean(loss1)
    
    d2_pos = tf.square(logits['E']-logits['B'])
    d2_neg = tf.square(logits['E']-logits['A'])
    loss2 = tf.maximum(0., margin + d2_pos - d2_neg)
    loss2 = tf.reduce_mean(loss2)    
    
    loss = loss1 + loss2 
    loss = tf.multiply(W_LB_RANK, loss, name='label_rank_loss')
    tf.add_to_collection('losses', loss)
    
    return loss
    
    
def triplet_fc_loss(fcs,labels):
    """ compute the triplet loss 
        fcs: a dict, fc output
        labels: a dict
    """
    ma = tf.cast(tf.equal(labels['S'], labels['E']),tf.float32) * NEG_MARGIN_FC_RANK
    mb = tf.cast(tf.not_equal(labels['S'], labels['E']),tf.float32) * POS_MARGIN_FC_RANK   
    margin =  ma + mb
    
    # S < A < B < E ; N 
    d1_pos = tf.reduce_sum(tf.square(fcs['S']-fcs['A']),1)
    d1_neg = tf.reduce_sum(tf.square(fcs['S']-fcs['B']),1)
    loss1 = tf.maximum(0., margin + d1_pos - d1_neg)
    loss1 = tf.reduce_mean(loss1)
    
    d2_pos = tf.reduce_sum(tf.square(fcs['E']-fcs['B']),1)
    d2_neg = tf.reduce_sum(tf.square(fcs['E']-fcs['A']),1)
    loss2 = tf.maximum(0., margin + d2_pos - d2_neg)
    loss2 = tf.reduce_mean(loss2)    

    loss = loss1 + loss2 
    loss = tf.multiply(W_FC_RANK , loss, name='feature_rank_loss')
    tf.add_to_collection('losses', loss)
    
    return loss
    
def sym_loss(fcs,fcs_flip):
    """ Compute the symmetry loss 
        fcs: a dict, fc layer
        fcs_flip: fc layer of flipped images
    """
#    loss_S = tf.reduce_sum(tf.square(fcs['S']-fcs_flip['S']),1)
    loss_A = tf.reduce_sum(tf.square(fcs['A']-fcs_flip['A']),1)
    loss_B = tf.reduce_sum(tf.square(fcs['B']-fcs_flip['B']),1)
#    loss_E = tf.reduce_sum(tf.square(fcs['E']-fcs_flip['E']),1)
    loss_N = tf.reduce_sum(tf.square(fcs['N']-fcs_flip['N']),1)
#    loss = loss_S + loss_A + loss_B + loss_E + loss_N 
    loss =  loss_A + loss_B + loss_N

    loss = tf.reduce_mean(loss)
    loss = tf.multiply(W_SYM , loss,name='symmetric')
    tf.add_to_collection('losses', loss)
    
    return loss
    
def neu_loss(fcs,labels):
    """ compute the nuetral loss. Different from nuetral
        fcs: a dict
    """
    ma = tf.cast(tf.equal(labels['S'], labels['E']),tf.float32) * NEG_MARGIN_NEU
    mb = tf.cast(tf.not_equal(labels['S'], labels['E']),tf.float32) * POS_MARGIN_NEU   
    margin =  ma + mb    
    
    d_NA = tf.reduce_sum(tf.square(fcs['A']-fcs['N']),1)
    d_NB = tf.reduce_sum(tf.square(fcs['B']-fcs['N']),1)
    
    loss_A = tf.maximum(0., margin - d_NA) 
    loss_B = tf.maximum(0., margin - d_NB)
    loss = loss_A + loss_B
    loss = tf.reduce_mean(loss)   
    loss = tf.multiply(W_NEU, loss, name='neutral_loss')
    tf.add_to_collection('losses', loss)
    
    return loss

def lb_round_loss(logits,labels):
    labels['S'] = tf.cast(labels['S'],tf.float32)
    labels['E'] = tf.cast(labels['E'],tf.float32)
    labels['A'] = tf.cast(labels['A'],tf.float32)
    labels['B'] = tf.cast(labels['B'],tf.float32)
    
# make predict close to an integer ?
    t_loss_S = tf.reduce_mean(tf.square(logits['S']-tf.round(logits['S'])))
    t_loss_E = tf.reduce_mean(tf.square(logits['E']-tf.round(logits['E'])))
    t_loss_A = tf.reduce_mean(tf.square(logits['A']-tf.round(logits['A'])))
    t_loss_B = tf.reduce_mean(tf.square(logits['B']-tf.round(logits['B'])))
    
    loss = tf.multiply(W_LB_ROUND , t_loss_S + t_loss_E + t_loss_A + t_loss_B, name='round_loss')
    tf.add_to_collection('losses', loss) 
    
    return loss


def lb_loss(logits,labels):
    """ compute the label loss.
        logits: a dict
    """

    labels['S'] = tf.cast(labels['S'],tf.float32)
    labels['E'] = tf.cast(labels['E'],tf.float32)
    labels['N'] = tf.cast(labels['N'],tf.float32)
    labels['A'] = tf.cast(labels['A'],tf.float32)
    labels['B'] = tf.cast(labels['B'],tf.float32)

    
    # convert task vector to indice
    depth = 5
    oneK = tf.one_hot(tf.squeeze(logits['T']['T']-1),depth,on_value=1.,off_value=0.)

    # compute the loss of corresponding elements 
    loss_S = tf.losses.mean_squared_error(tf.reduce_sum(tf.multiply(oneK,logits['S']),axis=1,keep_dims=True), labels['S'])
    loss_E = tf.losses.mean_squared_error(tf.reduce_sum(tf.multiply(oneK,logits['E']),axis=1,keep_dims=True), labels['E'])
#    loss_N = tf.losses.mean_pairwise_squared_error(tf.reduce_sum(tf.multiply(oneK,logits['N']),axis=1,keep_dims=True), labels['N'])
 

    loss = tf.multiply(W_LB , loss_S + loss_E, name='label_SE')
    tf.add_to_collection('losses', loss) 


#    loss_A = tf.losses.mean_squared_error(tf.reduce_sum(tf.multiply(oneK,logits['A']),axis=1,keep_dims=True), labels['A'])
#    loss_B = tf.losses.mean_squared_error(tf.reduce_sum(tf.multiply(oneK,logits['B']),axis=1,keep_dims=True), labels['B'])
#    
#    loss_AB = tf.add(loss_A,loss_B,name='label_AB')
#    tf.add_to_collection('losses', loss_AB) 
 
    
    return loss

def lb_loss2(logits,labels):
    """ compute the label loss.
        logits: a dict
    """

    labels['S'] = tf.cast(labels['S'],tf.float32)
    labels['E'] = tf.cast(labels['E'],tf.float32)
    labels['N'] = tf.cast(labels['N'],tf.float32)
    labels['A'] = tf.cast(labels['A'],tf.float32)
    labels['B'] = tf.cast(labels['B'],tf.float32)

    
    # convert task vector to indice
    depth = 5
    oneK = tf.one_hot(tf.squeeze(logits['T']['T']-1),depth,on_value=1.,off_value=0.)

    # compute the loss of corresponding elements 
    loss_S = tf.losses.mean_squared_error(tf.reduce_sum(tf.multiply(oneK,logits['S']),axis=1,keep_dims=True), labels['S'])
    loss_E = tf.losses.mean_squared_error(tf.reduce_sum(tf.multiply(oneK,logits['E']),axis=1,keep_dims=True), labels['E'])
#    loss_N = tf.losses.mean_pairwise_squared_error(tf.reduce_sum(tf.multiply(oneK,logits['N']),axis=1,keep_dims=True), labels['N'])
    

    loss = tf.multiply(W_LB , loss_S + loss_E, name='label_SE')
    tf.add_to_collection('losses', loss) 
    


    SUB = tf.cast(tf.squeeze(logits['SUB']['SUB']),tf.float32) 
    s_oneK = tf.reshape(tf.div(tf.sign(RC-SUB) + 1.0, 2.0),(-1,1))  # sign: choosen 1, not chosen : 0
    
    sum_s_oneK = tf.reduce_sum(s_oneK)
    
    preA = tf.reduce_sum(tf.multiply(oneK,logits['A']),axis=1,keep_dims=True)
    preB = tf.reduce_sum(tf.multiply(oneK,logits['B']),axis=1,keep_dims=True)
      

    loss_A = tf.reduce_sum(tf.multiply(s_oneK, tf.square(preA - labels['A']))) 
    loss_B = tf.reduce_sum(tf.multiply(s_oneK, tf.square(preB - labels['B']))) 
    loss_A = tf.cond(sum_s_oneK >0.0, lambda:loss_A/sum_s_oneK, lambda:0.0)
    loss_B = tf.cond(sum_s_oneK >0.0, lambda:loss_B/sum_s_oneK, lambda:0.0)
    
    loss_AB = tf.multiply(W_LBADD, loss_A + loss_B, name='label_AB')
    tf.add_to_collection('losses', loss_AB) 

    return loss
    

def add_loss_summary(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9,name='avg')
    losses = tf.get_collection('losses')
    loss_average_op = loss_averages.apply(losses+[total_loss])
    
    for l in losses+[total_loss]:
        tf.summary.scalar(l.op.name,l)
#        tf.summary.scalar(l.op.name,loss_averages.average(l))
    return loss_average_op

def train(total_loss,global_step):
    num_batches_per_epoch = NUM_EXAPLES_PER_EPOCH_FOR_TRAIN // BATCH_SIZE
     # number of batches per decay
    decay_steps = int(num_batches_per_epoch*NUM_EPOCHS_PER_DECAY)
    
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate',lr) 

    loss_average_op = add_loss_summary(total_loss)               
    
    # compute gradients
    with tf.control_dependencies([loss_average_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        
    apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
    
    # add histogram for trainable variables
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name,var)
        
    # add historgram for gradients 
    for grad,var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name+'/gradients',grad)
    
    # track the moving avarages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
                            MOVING_AVERAGE_DECAY,global_step)
    variable_average_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op,variable_average_op]):
        train_op = tf.no_op(name='train')
        
    return train_op
