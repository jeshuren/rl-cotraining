#!/usr/bin/python
# -*- coding: utf-8 -*-
import keras
from selfAttention import selfAttention

class biDirectionalGRU():

    def __init__(self, vocab_size, EMBEDDING_DIM, num_units, embedding_matrix):

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Embedding(input_dim=vocab_size,
                                         output_dim=EMBEDDING_DIM,
                                         mask_zero=False,
                                        weights = [embedding_matrix],trainable = False))
        self.model.add(keras.layers.Bidirectional(keras.layers.GRU(units=num_units,
                                                               return_sequences=True)))
        self.model.add(selfAttention(n_head = 1 , hidden_dim=num_units))
        self.model.add(keras.layers.Dense(units=2,activation='softmax'))
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        print self.model.summary()

    def fit(self,X,y,X_val,y_val):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,min_lr=0.0001)

        file_path = 'best_model_title.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', 
            save_best_only=True)

        callbacks = [reduce_lr,model_checkpoint]
        self.model.fit(X,y,validation_data=(X_val,y_val),epochs=10,callbacks = callbacks)

    def predict(self, X):
        self.model = keras.models.load_model('best_model_title.hdf5',custom_objects=selfAttention.get_custom_objects())
        return self.model.predict(X)
    
    def evaluate(self,X,y):
        return self.model.evaluate(X,y)[1]



