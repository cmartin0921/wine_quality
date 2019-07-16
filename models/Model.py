import tensorflow as tf
from time import time


class Model():

    def __init__(self, tensorboard=None):
        self._tensorboard = tensorboard
        self.model = None

    def create(self, input_dim=None, optimizer=None, seed=None):

        #     model = tf.keras.Sequential()
        #     model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu,
        #                                     input_dim=input_dim, use_bias=True))
        # #     model.add(tf.keras.layers.BatchNormalization())
        #     model.add(tf.keras.layers.Dropout(rate=0.2, seed=seed))
        #     model.add(tf.keras.layers.Dense(
        #         4, activation=tf.nn.relu, use_bias=True))
        # #     model.add(tf.keras.layers.BatchNormalization())
        #     model.add(tf.keras.layers.Dropout(rate=0.2, seed=seed))
        #     model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
        #     model.compile(optimizer='adam',
        #                   loss='binary_crossentropy', metrics=['accuracy'])

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu,
                                        input_dim=input_dim,
                                        use_bias=True)
                  )
        model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

        self.model = model

    def train(self, epochs=100, X_data=None, y_data=None, validation_data=[]):
        # tf.keras.backend.clear_session()
        history = self.model.fit(x=X_data, y=y_data,
                                 validation_data=validation_data,
                                 batch_size=128,
                                 epochs=epochs,
                                 shuffle=True,
                                 verbose=2,
                                 callbacks=[self._tensorboard])
        return history
