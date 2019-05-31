#!/usr/bin/python
# -*- coding: utf-8 -*-
import gym
from gym.utils import seeding
import tensorflow as tf
from biDirectionalGRU import biDirectionalGRU
from cnnClassifier import cnnClassifier
import numpy as np
import pandas as pd
import re
import json
import pickle as pkl
from keras.utils import to_categorical
import os
from gym import spaces
from sklearn.metrics import precision_recall_fscore_support

class CoTraining(gym.Env):

    def __init__(self):
        
        self.count = 0

        self.load_data_view1()       

        self.episode = 0
        
        self.load_data_view2()
        
        self.model1 = None
        self.model2 = None

        self.metrics_view1 = []
        self.metrics_view2 = []
        
        self.rep = pkl.load(open('data/rep.pkl'))
        
        self.members = pkl.load(open('data/members.pkl'))
        
        self.view1_rep_X = self.view1_unlabelled[self.rep,:]
        self.view2_rep_X = self.view2_unlabelled[self.rep,:]
        
        self.num_units = 100

        self.EMBEDDING_DIM = 100
        
        self.state = None

        self.seed()
        #self.reset()
        
        self.mode = 'train'
        self.action_space = spaces.Discrete(80)
        
    def toggleMode(self):
        if self.mode == 'train':
            self.mode = 'test'
        else:
            self.mode = 'train'

    def load_data_view1(self):
                
        self.view1_train_X = pkl.load(open('data/view1_train_X'))
        self.view1_train_y = pkl.load(open('data/view1_train_y'))
        
        self.view1_val_X = pkl.load(open('data/view1_val_X'))
        self.view1_val_y = pkl.load(open('data/view1_val_y'))
        
        self.view1_test_X = pkl.load(open('data/view1_test_X'))
        self.view1_test_y = pkl.load(open('data/view1_test_y'))
        
        self.view1_unlabelled = pkl.load(open('data/view1_unlabelled'))
        
        self.embedding_matrix1 = pkl.load(open('data/view1_embedding_matrix'))
        
        self.vocab_size1 = len(self.embedding_matrix1)
        
    def load_data_view2(self):
        
        self.view2_train_X = pkl.load(open('data/view2_train_X'))
        self.view2_train_y = pkl.load(open('data/view2_train_y'))
        
        self.view2_val_X = pkl.load(open('data/view2_val_X'))
        self.view2_val_y = pkl.load(open('data/view2_val_y'))
        
        self.view2_test_X = pkl.load(open('data/view2_test_X'))
        self.view2_test_y = pkl.load(open('data/view2_test_y'))
        
        self.view2_unlabelled = pkl.load(open('data/view2_unlabelled'))
        
        self.embedding_matrix2 = pkl.load(open('data/view2_embedding_matrix'))
        
        self.vocab_size2 = len(self.embedding_matrix2)
        
    
    def step(self, action):
        
        if self.mode == 'test':

            reward1 = self.model1.evaluate(self.view1_test_X, self.view1_test_y)
            reward2 = self.model2.evaluate(self.view2_test_X, self.view2_test_y)
            
        else:
            reward1 = self.model1.evaluate(self.view1_val_X, self.view1_val_y)
            reward2 = self.model2.evaluate(self.view2_val_X, self.view2_val_y)
            

        labels1 = np.argmax(self.model1.predict(self.view1_unlabelled[self.members[action]]),axis=-1) 
        self.view2_train_X = np.concatenate((self.view2_train_X,self.view2_unlabelled[self.members[action]]))
        self.view2_train_y = np.concatenate((self.view2_train_y,to_categorical(labels1,num_classes=2)))
        
        self.model2 = cnnClassifier(self.vocab_size2, self.EMBEDDING_DIM, self.embedding_matrix2)

        self.model2.fit(self.view2_train_X, self.view2_train_y,self.view2_val_X,self.view2_val_y)

        

        labels2 = np.argmax(self.model2.predict(self.view2_unlabelled[self.members[action]]),axis=-1)

        self.view1_train_X = np.concatenate((self.view1_train_X,self.view1_unlabelled[self.members[action]]))
        self.view1_train_y = np.concatenate((self.view1_train_y,to_categorical(labels2,num_classes=2)))
        
        self.model1 = biDirectionalGRU(self.vocab_size1, self.EMBEDDING_DIM, self.num_units, self.embedding_matrix1)
        self.model1.fit(self.view1_train_X, self.view1_train_y,self.view1_val_X,self.view1_val_y)


        if self.mode == 'test':
            reward1 = self.model1.evaluate(self.view1_test_X, self.view1_test_y) - reward1
            reward2 = self.model2.evaluate(self.view2_test_X, self.view2_test_y) - reward2
            
        else:
            reward1 = self.model1.evaluate(self.view1_val_X, self.view1_val_y) - reward1
            reward2 = self.model2.evaluate(self.view2_val_X, self.view2_val_y) - reward2

        if reward1 > 0 and reward2 > 0:
            reward = reward1 * reward2
        else:
            reward = 0
        
        tmp1 = self.model1.predict(self.view1_rep_X)
        tmp2 = self.model2.predict(self.view2_rep_X)

        self.state = np.hstack((tmp1,tmp2)).reshape(-1)

        info = {'reward1':reward1,
                'reward2':reward2,
                'action': action
                }
        
        done = False
        
        print info

        if reward == 0:
            self.count = self.count + 1
        
        if self.count > 5 or self.steps == 80:
            done = True
        
        if self.mode == 'test':
            v1 = np.argmax(self.model1.predict(self.view1_test_X), axis = -1)
            v2 = np.argmax(self.model2.predict(self.view2_test_X), axis = -1)
            m1 = precision_recall_fscore_support(np.argmax(self.view1_test_y,axis=-1), v1, average='macro')
            m2 = precision_recall_fscore_support(np.argmax(self.view2_test_y,axis=-1), v2, average='macro')

        else:
            v1 = np.argmax(self.model1.predict(self.view1_val_X), axis = -1)
            v2 = np.argmax(self.model2.predict(self.view2_val_X), axis = -1)
            m1 = precision_recall_fscore_support(np.argmax(self.view1_val_y,axis=-1), v1, average='macro')
            m2 = precision_recall_fscore_support(np.argmax(self.view2_val_y,axis=-1), v2, average='macro')

        self.steps = self.steps+1

        self.metrics_view1[self.steps,:] = m1[:3]
        self.metrics_view2[self.steps,:] = m2[:3]

        return (self.state, reward, done, info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.count = 0

        self.load_data_view1()       
        
        self.load_data_view2()

        if self.metrics_view1 != []:
            np.savetxt("results/metrics_view1_"+self.mode+"_"+str(self.episode)+".csv", self.metrics_view1, delimiter=",")
            np.savetxt("results/metrics_view2_"+self.mode+"_"+str(self.episode)+".csv", self.metrics_view2, delimiter=",")

        self.metrics_view1 = np.zeros((80,3))
        self.metrics_view2 = np.zeros((80,3))

        self.steps = 0

        self.model1 = biDirectionalGRU(self.vocab_size1, self.EMBEDDING_DIM, self.num_units, self.embedding_matrix1)
        self.model2 = cnnClassifier(self.vocab_size2, self.EMBEDDING_DIM, self.embedding_matrix2)

        self.model1.fit(self.view1_train_X, self.view1_train_y, self.view1_val_X, self.view1_val_y)
        self.model2.fit(self.view2_train_X, self.view2_train_y, self.view2_val_X, self.view2_val_y)

        tmp1 = self.model1.predict(self.view1_rep_X)
        tmp2 = self.model2.predict(self.view2_rep_X)

        self.state = np.hstack((tmp1,tmp2)).reshape(-1)

        self.done = False

        self.episode = self.episode + 1
        return self.state