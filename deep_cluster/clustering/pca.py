import tensorflow as tf
import tensorflow_probability as tfp


class PCA:
    def __init__(self, data, k):
        self.y = self.fit_transform(data, k)

    @staticmethod
    def __compute_cov(x):
        mean_x = tf.reduce_mean(x, axis=0)
        mx = tf.matmul(tf.transpose(mean_x), mean_x)
        vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float32)
        cov_xx = vx - mx
        return cov_xx

    @staticmethod
    def __get_k_eigen_vector(x, k):
        covariance_matrix = tfp.stats.covariance(x)
        # covariance_matrix = PCA.__compute_cov(x)
        s, u, v = tf.linalg.svd(covariance_matrix)
        return u[:, :k]

    @staticmethod
    def __project_dimension(x, eigen_vector):
        return tf.tensordot(tf.transpose(eigen_vector), tf.transpose(x), axes=1)

    def fit_transform(self, x, k):
        eigen_vector = PCA.__get_k_eigen_vector(x, k)
        return tf.transpose(PCA.__project_dimension(x, eigen_vector))
