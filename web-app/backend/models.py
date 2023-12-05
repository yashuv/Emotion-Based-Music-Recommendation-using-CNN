import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Layer, Conv2D, BatchNormalization, GlobalAveragePooling2D,AveragePooling2D



class CNNModel(Layer):
    def __init__(self, out_channels, kernel_size=3):
        super(CNNModel, self).__init__()
        self.conv = Conv2D(out_channels, kernel_size, padding="same")
        self.bn = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class ResBlock(Layer):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.cnn1 = CNNModel(channels[0])
        self.cnn2 = CNNModel(channels[1])
        self.cnn3 = CNNModel(channels[2])
        self.pooling = MaxPooling2D()
        self.identity_mapping = Conv2D(channels[1], 3, padding="same")

    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor, training=training)
        x = self.cnn2(x, training=training)
        x = self.cnn3(x + self.identity_mapping(input_tensor),
                      training=training)
        return self.pooling(x)


class ResNet_Like(Model):
    def __init__(self, num_classes):
        super(ResNet_Like, self).__init__()
        self.block1 = ResBlock([32, 32, 64])
        self.block2 = ResBlock([128, 128, 256])
        self.block3 = ResBlock([128, 256, 512])
        self.pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.pool(x)
        return self.classifier(x)

    def model(self):
        x = tf.keras.Input(shape=(48, 48, 3))
        return Model(inputs=[x], outputs=self.call(x))