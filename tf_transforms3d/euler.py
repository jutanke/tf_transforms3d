import tensorflow as tf
import math as m


def similarity(r1, r2):
    """
    calculates how similar to batches of euler angles are
    :param r1: (n_batch x 3)
    :param r2: (n_batch x 3)
    """
    PI = tf.constant(m.pi)
    r1 = tf.reshape(r1, (-1, 1))
    r2 = tf.reshape(r2, (-1, 1))
    d = r1 - r2
    distance = tf.math.mod((d+PI), 2 * PI) - PI
    distance = tf.reshape(distance, (-1, 3))
    return distance


def euler2mat(r):
    """
    we only support sxyz
    convert a batch of euler angles to a rotation matrix
    :param r: (n_batch x 3)
    """
    n_batch = tf.shape(r)[0]
    const0 = tf.zeros((n_batch,))
    const1 = tf.ones((n_batch,))
    X = r[:, 0]
    Y = r[:, 1]
    Z = r[:, 2]
    X_cos = tf.cos(X)
    X_sin = tf.sin(X)
    r1 = tf.stack([const1, const0, const0], axis=1)
    r2 = tf.stack([const0,  X_cos, -X_sin], axis=1)
    r3 = tf.stack([const0,  X_sin,  X_cos], axis=1)
    Rx = tf.stack([r1, r2, r3], axis=1)
    Y_cos = tf.cos(Y)
    Y_sin = tf.sin(Y)
    r1 = tf.stack([Y_cos,  const0,  Y_sin], axis=1)
    r2 = tf.stack([const0, const1, const0], axis=1)
    r3 = tf.stack([-Y_sin, const0,  Y_cos], axis=1)
    Ry = tf.stack([r1, r2, r3], axis=1)
    Z_cos = tf.cos(Z)
    Z_sin = tf.sin(Z)
    r1 = tf.stack([ Z_cos, -Z_sin, const0], axis=1)
    r2 = tf.stack([ Z_sin,  Z_cos, const0], axis=1)
    r3 = tf.stack([const0, const0, const1], axis=1)
    Rz = tf.stack([r1, r2, r3], axis=1)
    Rzy = tf.matmul(Rz, Ry)
    R = tf.matmul(Rzy, Rx)
    # R = tf.transpose(R, [0, 2, 1])
    return R


def mat2euler(M):
    """
    we only support ('sxyz': (0, 0, 0, 0))
    :param M: (n_batch x 3 x 3)
    """
    eps = tf.constant(0.00001)
    i = 0
    j = 1
    k = 2
    cy = tf.sqrt(M[:, i, i] * M[:, i, i] + M[:, j, i] * M[:, j, i])

    ax = tf.where(tf.greater_equal(cy, eps), 
                  tf.atan2(M[:, k, j], M[:, k, k]),
                  tf.atan2(-M[:, j, k], M[:, j, j]))
    ay = tf.atan2(-M[:, k, i], cy)
    az = tf.where(tf.greater_equal(cy, eps),
                  tf.atan2(M[:, j, i], M[:, i, i]),
                  .0)

    return ax, ay, az
