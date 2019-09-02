# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier

class HCN(object):

    def __init__(self, action_mask_dict, obs_size, act_size, hidden_dim=128, lr=0.1):
        self.obs_size = obs_size
        self.act_size = act_size
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.init_c_ = np.zeros([1,hidden_dim],dtype=np.float32)
        self.init_h_ = np.zeros([1,hidden_dim],dtype=np.float32)
        self.action_mask_dict = action_mask_dict
        self.build_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action_mask(self, feat_ctx):
        action_mask = np.zeros([self.act_size],dtype=np.float32)
        indices = self.action_mask_dict[''.join(map(str,feat_ctx))]
        for index in indices:
            action_mask[index-1] = 1.
        return action_mask

    def build_model(self):
        self.feats = tf.placeholder(tf.float32,[1,self.obs_size],name='input_feats')
        self.init_c = tf.placeholder(tf.float32,[1,self.hidden_dim])
        self.init_h = tf.placeholder(tf.float32,[1,self.hidden_dim])
        self.action = tf.placeholder(tf.int32,name='real_action')
        self.action_mask = tf.placeholder(tf.float32,[self.act_size],name='action_mask')
        Wi = tf.get_variable('Wi',[self.obs_size,self.hidden_dim],initializer=xavier())
        bi = tf.get_variable('bi',[self.hidden_dim],initializer=tf.constant_initializer(0.))
        projected = tf.matmul(self.feats,Wi)+bi 
        lstm = tf.contrib.rnn.LSTMCell(self.hidden_dim,state_is_tuple=True)
        lstm_op,self.state = lstm(inputs=projected,state=(self.init_c,self.init_h))
        reshaped = tf.concat(axis=1,values=(self.state.c,self.state.h))
        Wo = tf.get_variable('Wo',[2*self.hidden_dim,self.act_size],initializer=xavier())
        bo = tf.get_variable('bo',[self.act_size],initializer=tf.constant_initializer(0.))
        self.logits = tf.matmul(reshaped,Wo)+bo
        self.probs = tf.multiply(tf.squeeze(tf.nn.softmax(self.logits)),self.action_mask)
        self.pred = tf.arg_max(self.probs,dimension=0)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.action) #??
        self.train_op = tf.train.AdadeltaOptimizer(self.lr).minimize(self.loss)

    def train_step(self, feats, action, action_mask):
        _,loss,state_c,state_h = self.sess.run([
            self.train_op,self.loss,self.state.c,self.state.h],
            feed_dict = {
                self.feats:feats.reshape([1,self.obs_size]),
                self.action:[action],
                self.init_c:self.init_c_,
                self.init_h:self.init_h_,
                self.action_mask:action_mask})
        self.init_c_ = state_c
        self.init_h_ = state_h
        return loss

    def predict_action(self, feats, action_mask):
        probs,pred,state_c,state_h = self.sess.run([
            self.probs,self.pred,self.state.c,self.state.h], 
            feed_dict = { 
                self.feats:feats.reshape([1,self.obs_size]), 
                self.init_c:self.init_c_,
                self.init_h:self.init_h_,
                self.action_mask:action_mask})
        self.init_c_ = state_c
        self.init_h_ = state_h
        return pred

    def reset_state(self):
        self.init_c_ = np.zeros([1,self.hidden_dim],dtype=np.float32)
        self.init_h_ = np.zeros([1,self.hidden_dim],dtype=np.float32)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess,'ckpt/hcn.ckpt',global_step=0)

    def load(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('ckpt/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess,ckpt.model_checkpoint_path)

if __name__ == '__main__':
    pass
