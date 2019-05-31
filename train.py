#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gym
import gym_cotraining

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'cotraining-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)


# In[ ]:


nb_actions = 80

# Next, we build a very simple model.
from keras.layers import Input, Embedding,Dense, Concatenate, Conv1D, Lambda, Reshape
from keras.models import Model
import keras.backend as K


inp = Input((1,320))
rs = Reshape((80,4))(inp)
unstacked = Lambda(lambda x: K.tf.unstack(x, axis=1))(rs)
common_dense = Dense(3,activation='relu')
dense_outputs = [common_dense(x) for x in unstacked]

concat = Concatenate(axis=-1)(dense_outputs)
dense2 = Dense(128, activation='relu')(concat)
out = Dense(80, activation='linear')(dense2)

model = Model(inputs=inp, outputs= out)
print model.summary()

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=10000, nb_max_episode_steps = 80, visualize=False, verbose=1)


# In[ ]:


# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)


# In[ ]:


env.toggleMode()


# In[ ]:


# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=2, visualize=False)


