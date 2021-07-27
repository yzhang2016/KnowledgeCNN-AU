#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:58:06 2017

@author: yong
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:04:47 2017

@author: yong
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:57:58 2017

@author: yong
"""

import time 
import tensorflow as tf
import numpy as np 
import ShollowCNN_xavier as cnnmodel
import bp4d_input_weak as bp4d_input
import ShollowCNN_AUModel_weak as AU_model
import variables as var
from subprocess import call
import shlex 
#import sklearn.metrics as skmet
#import scipy.stats.stats as scystat

RC = var.RC

PRE_FIX_NAME = 'weak_AU_MT_s32_part{0}'.format(RC)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir','../../data/tupleFileNames_MT/',
                           """ files""")
tf.app.flags.DEFINE_string('data_eval_dir','../../data/singleFileNames/',
                           """ files""")
tf.app.flags.DEFINE_string('train_dir','../log/{0}/train_log'.format(PRE_FIX_NAME),
                           """Directory for log file""")
tf.app.flags.DEFINE_string('saver_dir','../log/{0}/saved_model'.format(PRE_FIX_NAME),
                           """Directory for log file""")
tf.app.flags.DEFINE_integer('max_steps', 48000,
                            """Number of batches to run""")
tf.app.flags.DEFINE_bool('log_device_placement',False,
                         """wheter to log device placement""")
tf.app.flags.DEFINE_integer('log_frequency',10,
                            """How often to log results to the console""")


#tf.app.flags.DEFINE_bool('train_mode',True,
#                         """The training mode """)
train_log = '../log/{0}/train_log/'.format(PRE_FIX_NAME)
saver_dir = '../log/{0}/saved_model/model.ckpt'.format(PRE_FIX_NAME)

AUIND = 0

def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
        
    return session


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        
        train_mode = tf.constant(True,dtype=tf.bool)
        eval_mode = tf.constant(False,dtype=tf.bool)
        
        # generate training batch 
        imgs,flip_imgs, lbs,nms, tsk,sub = bp4d_input.distorted_inputs(FLAGS.data_dir,AU_model.BATCH_SIZE)
        eval_imgs, eval_lbs,eval_nms = bp4d_input.inputs_eval(FLAGS.data_eval_dir,AU_model.BATCH_SIZE,evalShuffle=True)
        
        
        # build graph and compute logits
        SHCNN = cnnmodel.ShollowCNN(dropout=1.0)
        logit_S, fc_S = SHCNN.inference(imgs['S'],train_mode=train_mode,is_build=True)
        logit_A, fc_A = SHCNN.inference(imgs['A'],train_mode=train_mode,is_build=False)
        logit_B, fc_B = SHCNN.inference(imgs['B'],train_mode=train_mode,is_build=False)
        logit_E, fc_E = SHCNN.inference(imgs['E'],train_mode=train_mode,is_build=False)
        logit_N, fc_N = SHCNN.inference(imgs['N'],train_mode=train_mode,is_build=False)
        
        _, fc_flip_S = SHCNN.inference(flip_imgs['S'],train_mode=train_mode,is_build=False)
        _, fc_flip_A = SHCNN.inference(flip_imgs['A'],train_mode=train_mode,is_build=False)
        _, fc_flip_B = SHCNN.inference(flip_imgs['B'],train_mode=train_mode,is_build=False)
        _, fc_flip_E = SHCNN.inference(flip_imgs['E'],train_mode=train_mode,is_build=False)
        _, fc_flip_N = SHCNN.inference(flip_imgs['N'],train_mode=train_mode,is_build=False)      
        
        logits = {
                'S': logit_S,
                'A': logit_A,
                'B': logit_B,
                'E': logit_E,
                'N': logit_N,
                'T': tsk, 
                'SUB':sub}
        
        fcs = {
                'S': fc_S,
                'A': fc_A,
                'B': fc_B,
                'E': fc_E,
                'N': fc_N} 
        
        fcs_flip = {
                'S': fc_flip_S,
                'A': fc_flip_A,
                'B': fc_flip_B,
                'E': fc_flip_E,
                'N': fc_flip_N}    
        
        # compute loss 
        loss = AU_model.comp_loss(fcs,fcs_flip,logits,lbs)
        
        # evaluation
        eval_logits, _= SHCNN.inference(eval_imgs,train_mode=eval_mode,is_build=False)
        eval_MSE, eval_MAE = AU_model.loss_eval(eval_logits,eval_lbs)
        tf.summary.scalar('eval_loss_MSE',eval_MSE)
        tf.summary.scalar('eval_loss_MAE',eval_MAE)
        
        # train the model
        train_op = AU_model.train(loss,global_step)
        
        saver = tf.train.Saver(max_to_keep=5)
        
        class _LoggerHook(tf.train.SessionRunHook):
            """log loss and runtime"""
            def begin(self):
                self._step = -1 
                self._start_time = time.time()
                
            def before_run(self,run_context):
                self._step += 1
                return tf.train.SessionRunArgs([loss,eval_MSE,eval_MAE,logit_A,lbs['A'],eval_logits,eval_lbs,tsk['T']])
                
            def after_run(self,run_context,run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time
                    
                    loss_value,MSE,MAE,logit,label, eval_logits, eval_lbs, TSK= run_values.results

                    
                    temlogit = np.reshape( [logit[i,j] for i,j in zip(range(5),TSK-1)],(-1,1)).astype(np.float32)
                    temlabel = np.reshape(label[0:5,:], (-1,1)).astype(np.float32)
                    temevallog = np.reshape([eval_logits[i,j] for i,j in zip(range(5),TSK-1)],(-1,1)).astype(np.float32)
                    temevallb = np.reshape([eval_lbs[i,j] for i,j in zip(range(5),TSK-1)], (-1,1)).astype(np.float32)

                    
                    print(np.concatenate((temlogit,temlabel,temevallog,temevallb),1))
                    
                    examples_per_sec = FLAGS.log_frequency * AU_model.BATCH_SIZE / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)
                    
                    format_str = ('Step %d, loss=%.5f,' 
                                  ' MAE=%.3f, MSE=%.3f'
                                    '(%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (self._step, loss_value,
                                        MAE, MSE,
                                        examples_per_sec, sec_per_batch))
                        
                    
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(gpu_options=gpu_options,
                                      log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            print(SHCNN.get_var_count())
            
            ckpt = tf.train.get_checkpoint_state(train_log)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(mon_sess,ckpt.model_checkpoint_path)
#                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            
            
            cnt = -1 #62621
            while not mon_sess.should_stop():
                mon_sess.run(train_op) 
                cnt += 1
                intv = 300
                if cnt % (intv) == 0:
                    saver.save(get_session(mon_sess),saver_dir,global_step=cnt)
#                    
                    call(shlex.split('sh writeIteration.sh {0}'.format(cnt//intv)))
                    call(shlex.split('python ShollowCNN_AUModel_eval_weak.py'))
    
def main(argv=None):
    if not tf.gfile.Exists(FLAGS.train_dir):
#        raise ValueError('dir already exists')
        tf.gfile.MakeDirs(FLAGS.train_dir)
    
    
    if not tf.gfile.Exists(FLAGS.saver_dir):
        tf.gfile.MakeDirs(FLAGS.saver_dir)
    train()

if __name__=='__main__':
    tf.app.run()
