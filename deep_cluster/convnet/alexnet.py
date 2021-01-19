import tensorflow as tf

from deep_cluster.preprocessing.dataset import Dataset


class AlexNet:
    SEED = 42

    def __init__(self):
        self.initializer = tf.initializers.glorot_uniform(seed=AlexNet.SEED)

        self.conv_shapes = (
            [11, 11, 1, 96],
            [5, 5, 96, 256],
            [3, 3, 256, 384],
            [3, 3, 384, 384],
            [3, 3, 384, 256],
        )

        self.conv_filters = []
        for i in range(len(self.conv_shapes)):
            self.conv_filters.append(self.__build_weights(self.conv_shapes[i], 'conv_weight{}'.format(i)))

        self.fc_weights_shapes = (
            [9216, 4096],
            [4096, 4096],
            [4096, 4096]
        )

        self.fc_weights = []
        for i in range(len(self.fc_weights_shapes)):
            self.fc_weights.append(self.__build_weights(self.fc_weights_shapes[i], 'fc_weight{}'.format(i)))

    @staticmethod
    def __conv2d(inputs, filters, stride, padding="VALID"):
        out = tf.nn.conv2d(inputs, filters, strides=[1, stride, stride, 1], padding=padding)
        mean, variance = tf.nn.moments(out, axes=0)
        out = tf.nn.batch_normalization(out, mean, variance, offset=0, scale=0, variance_epsilon=1e-20)
        return tf.nn.relu(out)

    @staticmethod
    def __maxpool(inputs, pool_size, stride):
        return tf.nn.max_pool2d(inputs, ksize=[1, pool_size, pool_size, 1], padding="VALID",
                                strides=[1, stride, stride, 1])

    @staticmethod
    def __dense(inputs, weights, dropout_rate=.5):
        out = tf.nn.dropout(inputs, rate=dropout_rate)
        return tf.nn.relu(tf.matmul(out, weights))

    def __build_weights(self, shape, name):
        return tf.Variable(initial_value=self.initializer(shape), name=name, trainable=True, dtype=tf.float32)

    def build_model(self, x):
        x = tf.cast(x, dtype=tf.float32)

        c1 = AlexNet.__conv2d(x, self.conv_filters[0], stride=4)
        p1 = AlexNet.__maxpool(c1, pool_size=3, stride=2)

        c2 = AlexNet.__conv2d(p1, self.conv_filters[1], stride=1, padding="SAME")
        p2 = AlexNet.__maxpool(c2, pool_size=3, stride=2)

        c3 = AlexNet.__conv2d(p2, self.conv_filters[2], stride=1, padding="SAME")

        c4 = AlexNet.__conv2d(c3, self.conv_filters[3], stride=1, padding="SAME")

        c5 = AlexNet.__conv2d(c4, self.conv_filters[4], stride=1, padding="SAME")

        p3 = AlexNet.__maxpool(c5, pool_size=3, stride=2)

        p3_reshaped = tf.reshape(p3, [-1, 9216])

        fc1 = AlexNet.__dense(p3_reshaped, self.fc_weights[0])
        fc2 = AlexNet.__dense(fc1, self.fc_weights[1])

        fc3 = tf.matmul(fc2, [4096, 4096])
