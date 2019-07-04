import tensorflow as tf
import math as m


def mat2quat(M):
    """
    we only support sxyz
    :param M: (n_batch x 3 x 3)
    """
    batchsize = tf.shape(M)[0]
    Qxx = M[:, 0, 0]
    Qyx = M[:, 0, 1]
    Qzx = M[:, 0, 2]
    Qxy = M[:, 1, 0]
    Qyy = M[:, 1, 1]
    Qzy = M[:, 1, 2]
    Qxz = M[:, 2, 0]
    Qyz = M[:, 2, 1]
    Qzz = M[:, 2, 2]

    zero = tf.zeros(shape=(batchsize, ))

    r1 = tf.stack([Qxx - Qyy - Qzz,            zero,            zero,            zero])
    r2 = tf.stack([      Qyx + Qxy, Qyy - Qxx - Qzz,            zero,            zero])
    r3 = tf.stack([      Qzx + Qxz,       Qzy + Qyz, Qzz - Qxx - Qyy,            zero])
    r4 = tf.stack([      Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx, Qxx + Qyy + Qzz])

    K = tf.stack([r1, r2, r3, r4])
    K = tf.transpose(K, (2, 0, 1))
    
    K = K / 3.

    vals, vecs = tf.linalg.eigh(K)
    largest_eigv = 3
    q0 = vecs[:, 3, largest_eigv]
    qi = vecs[:, 0, largest_eigv]
    qj = vecs[:, 1, largest_eigv]
    qk = vecs[:, 2, largest_eigv]

    # q * -1 and q correspond to same rotation
    # wow.. this is ugly...
    qi = tf.where(tf.greater_equal(q0, 0.), qi, -qi)
    qj = tf.where(tf.greater_equal(q0, 0.), qj, -qj)
    qk = tf.where(tf.greater_equal(q0, 0.), qk, -qk)
    q0 = tf.where(tf.greater_equal(q0, 0.), q0, -q0)

    Q = tf.transpose(tf.stack([q0, qi, qj, qk]))

    return Q
