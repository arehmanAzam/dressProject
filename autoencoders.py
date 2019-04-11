import tensorflow as tf
import keras

import keras.layers as L



class HeadAutoEncoder:
    """
    autoencoder for head representation
    """
    def __init__(self,input_shape=(88, 88, 3),code_size=128):
        self.input_shape=input_shape
        self.code_size=code_size
    def model_func(self):
        encoder = keras.models.Sequential()
        encoder.add(L.InputLayer(self.input_shape))
        encoder.add(L.Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        encoder.add(L.MaxPool2D(pool_size=(2, 2)))
        encoder.add(L.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        encoder.add(L.MaxPool2D(pool_size=(2, 2)))
        encoder.add(L.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        encoder.add(L.MaxPool2D(pool_size=(2, 2)))
        encoder.add(L.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        encoder.add(L.MaxPool2D(pool_size=(2, 2)))
        encoder.add(L.Flatten())
        encoder.add(L.Dense(self.code_size))

        # decoder
        decoder = keras.models.Sequential()
        decoder.add(L.InputLayer((self.code_size,)))
        decoder.add(L.Dense(1024))
        decoder.add(L.Reshape((2, 2, 256)))
        decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='valid'))
        decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='valid'))
        decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation=None, padding='same'))

        inp = L.Input(self.input_shape)
        code = encoder(inp)
        reconstruction = decoder(code)

        autoencoder = keras.models.Model(inp, reconstruction)
        autoencoder.compile('adamax', 'mse')

        return encoder, decoder,autoencoder

class BodyAutoEncoder:
    """
    autoencoder for head representation
    """
    def __init__(self,input_shape=(128, 128, 3),code_size=256):
        self.input_shape=input_shape
        self.code_size=code_size
    def model_func(self):
        encoder = keras.models.Sequential()
        encoder.add(L.InputLayer(self.input_shape))
        encoder.add(L.Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        encoder.add(L.MaxPool2D(pool_size=(2, 2)))
        encoder.add(L.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        encoder.add(L.MaxPool2D(pool_size=(2, 2)))
        encoder.add(L.Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        encoder.add(L.MaxPool2D(pool_size=(2, 2)))
        encoder.add(L.Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        encoder.add(L.MaxPool2D(pool_size=(2, 2)))
        encoder.add(L.Flatten())
        encoder.add(L.Dense(self.code_size))

        # decoder
        decoder = keras.models.Sequential()
        decoder.add(L.InputLayer((self.code_size,)))
        decoder.add(L.Dense(1024))
        decoder.add(L.Reshape((2, 2, 256)))
        decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))
        decoder.add(L.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation=None, padding='same'))

        inp = L.Input(self.input_shape)
        code = encoder(inp)
        reconstruction = decoder(code)

        autoencoder = keras.models.Model(inp, reconstruction)
        autoencoder.compile('adamax', 'mse')
        return encoder, decoder,autoencoder




if __name__ == '__main__':
    head_coder=HeadAutoEncoder()
    head_coder.model_func()