#!/usr/bin/python
# -*- coding: utf-8 -*-
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, Dropout
from keras.models import Model
import keras

class cnnClassifier():

    def __init__(self, vocab_size, EMBEDDING_DIM, embedding_matrix):

        embedding_layer = Embedding(vocab_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=200,
                            trainable=True)

        # applying a more complex convolutional approach
        convs = []
        filter_sizes = [3,4,5]

        sequence_input = Input(shape=(200,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        for fsz in filter_sizes:
            l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)
            l_pool = MaxPooling1D(5)(l_conv)
            convs.append(l_pool)

        l_merge = Concatenate(axis=1)(convs)
        l_cov1= Conv1D(128, 5, activation='relu')(l_merge)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        l_flat = Flatten()(l_pool1)
        l_dense = Dense(128, activation='relu')(l_flat)
        preds = Dense(2, activation='softmax')(l_dense)

        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        print("model fitting - more complex convolutional neural network")
        print self.model.summary()

    def fit(self,X,y,X_val,y_val):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,min_lr=0.0001)

        file_path = 'best_model_passage.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', 
            save_best_only=True)

        callbacks = [reduce_lr,model_checkpoint]
        self.model.fit(X,y,validation_data=(X_val,y_val),epochs=10,callbacks = callbacks)

    def predict(self, X):
        self.model = keras.models.load_model('best_model_passage.hdf5')
        return self.model.predict(X)
    
    def evaluate(self,X,y):
        return self.model.evaluate(X,y)[1]



