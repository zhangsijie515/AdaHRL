import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten


class ControllerNet(tf.keras.Model):
    def __init__(self, input_shape):
        super(ControllerNet, self).__init__()
        self.input_layer = Input(input_shape)
        self.f1 = Flatten()
        self.d1 = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.d2 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.d3 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.d4 = Dense(4)
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=None, mask=None):
        x_1 = self.f1(inputs)
        x_2 = self.d1(x_1)
        x_3 = self.d2(x_2)
        x_4 = self.d3(x_3)
        y = self.d4(x_4)
        return y


class MetaControllerNet(tf.keras.Model):
    def __init__(self, input_shape):
        super(MetaControllerNet, self).__init__()
        self.input_layer = Input(input_shape)
        self.d1 = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.d2 = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2())
        self.d3 = Dense(input_shape)
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=None, mask=None):
        x_1 = self.d1(inputs)
        x_2 = self.d2(x_1)
        y = self.d3(x_2)
        return y
