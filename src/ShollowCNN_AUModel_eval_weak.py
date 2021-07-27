#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:46:08 2017

@author: yong
"""


import numpy as np 
import tensorflow as tf 
import bp4d_input_weak as bp4d_input
import ShollowCNN_xavier as cnnmodel
import ShollowCNN_AUModel_weak as AU_model
import pickle
import scipy.stats as stats
import sklearn.metrics as skmetric
import scipy.io as scipIO
import iters as iters
import variables as var

RC = var.RC

BATCH_SIZE = AU_model.BATCH_SIZE

FLAGS = tf.app.flags.FLAGS

PRE_FIX_NAME = 'weak_AU_MT_s32_part{0}'.format(RC)
#PRE_FIX_NAME = 'weak_AU_MT_s32_part8.0_1000'


tf.app.flags.DEFINE_string('eval_dir','../log/{0}/evalRes/'.format(PRE_FIX_NAME),
                           """path to save the evaluation file""")
tf.app.flags.DEFINE_string('pre_dir','../log/{0}/preRes/'.format(PRE_FIX_NAME),
                           """save the predictions""")
tf.app.flags.DEFINE_string('data_dir','../../data/singleFileNames/',
                           """ files""")
tf.app.flags.DEFINE_string('checkpoint_dir','../log/{0}/train_log'.format(PRE_FIX_NAME),
                           """path contain checkpoint file""")
tf.app.flags.DEFINE_integer('num_eval', 75000,
                            """number of testing samples""")
tf.app.flags.DEFINE_bool('eval',True,
                            """Evaluation""")

if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
#    tf.gfile.DeleteRecursively(FLAGS.eval_dir)

intv = 300

global_step = intv*  iters.iter #900 * 111
#global_step = 26000 #900 * 111


with tf.Graph().as_default() as g:
    train_mode = tf.constant(False,dtype=tf.bool)
    images,labels,names = bp4d_input.inputs_eval(FLAGS.data_dir,BATCH_SIZE,evalShuffle=False)

    SHCNN = cnnmodel.ShollowCNN()
    logits, feats = SHCNN.inference(images,train_mode=train_mode,is_build=True)  
 
    
    eval_MSE, eval_MAE = AU_model.loss_eval(logits,labels)
    
    
#    var_avg = tf.train.ExponentialMovingAverage(AU_model.MOVING_AVERAGE_DECAY)
#    var_to_restore = var_avg.variables_to_restore()
#    saver = tf.train.Saver(var_to_restore)
#    
    saver = tf.train.Saver()
    
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
#        if ckpt and ckpt.model_checkpoint_path:
#            saver.restore(sess,ckpt.model_checkpoint_path)
#            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
     
        saver.restore(sess,'../log/{0}/saved_model/model.ckpt-{1}'.format(PRE_FIX_NAME,global_step))
        coord = tf.train.Coordinator()
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess,coord=coord,daemon=True,start=True))

#        
        num_iter = FLAGS.num_eval // BATCH_SIZE
        step = 0
        tsPreLbs = []
        tsNames = []
        tsLabels = []
        predFeats = [] 
        while step < num_iter:
            preLb,Feat,Labels,Names,mse,mae = sess.run([logits,feats,labels,names,eval_MSE, eval_MAE])
 
#            Labels = np.reshape(Labels, (-1,bp4d_input.NUM_CLASSES)).astype(np.float32)
              
            tsLabels.extend(Labels)
            tsPreLbs.extend(preLb)
            tsNames.append(np.reshape(Names,[-1,1]))
            
            
            predFeats.extend(Feat)
            
            step += 1
            print('bath %003d : MSE = %.3f, MAE = %.3f '  %(step, mse, mae) )
        
        tsLabels = np.array(tsLabels).astype(np.float32)
        tsPreLbs = np.array(tsPreLbs).astype(np.float32)
        tsNames = np.array(tsNames)
        predFeats = np.concatenate(predFeats,axis=0)
        
        predFeats = np.reshape(predFeats,(-1,128))
        scipIO.savemat('../log/{0}/evalRes/evalRes-{1}.mat'.format(PRE_FIX_NAME,global_step),
                       {'orgLB': tsLabels, 'predLB':tsPreLbs, 'feat':predFeats,'names': tsNames})


        PCC = [stats.pearsonr(tsLabels[:,i],tsPreLbs[:,i])[0] for i in range(0,5)]
        MAE = np.mean(np.abs(tsLabels-tsPreLbs),axis=0)
        MSE = np.mean(np.square(tsLabels-tsPreLbs),axis=0)

        print('PCC=', PCC)
        print('MAE=', MAE)
        print('MSE=', MSE)
        
        print('mean: PCC=%f, MAE=%f, MSE=%f' %(np.mean(PCC),np.mean(MAE),np.mean(MSE)))

#        data = {'name':tsNames,
#                'label':tsLabels,
#                'prediction':tsPreLbs}
#        with open('./log/ShollowCNN/preRes/evalRes.pickle','wb') as f:
#            pickle.dump(data,f)
#        
#        summary = tf.Summary()
        #summary.ParseFromString(sess.run(summary_op))
        #summary.value.add(tag='Precision', simple_value=tsAcc)
#        summary_writer.add_summary(summary,global_step)
        
        coord.request_stop()
        coord.join(threads,stop_grace_period_secs=10)

    

       # A = pickle.load(open('/home/yong/Desktop/yong/code/log/ShallowCNN/preRes/evalRes.pickle','rb'))
        
        
        
        
        
        
        
        
        
