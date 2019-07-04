import tensorflow as tf
import math as m


def qmult(q1, q2):
    """
    mutiply two batches of quaternions element-wise
    :param q1: [n_batch x 4]
    :param q2: [n_batch x 4]
    """
    w1 = q1[:, 0]; x1 = q1[:, 1]; y1 = q1[:, 2]; z1 = q1[:, 3]
    w2 = q2[:, 0]; x2 = q2[:, 1]; y2 = q2[:, 2]; z2 = q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    q = tf.transpose(tf.stack([w, x, y, z]))
    return q


def quat2mat(Q):
    """
    we only support sxyz
    :param Q: (n_batch x 4)
    """
    w = Q[:, 0]
    x = Q[:, 1]
    y = Q[:, 2]
    z = Q[:, 3]
    Nq = w*w + x*x + y*y + z*z

    Nq_is_zero = tf.less_equal(Nq, 0.0000001)

    s = 2.0 / Nq  # division by zero IS evaluated (but we fix it later!)
    X = x * s
    Y = y * s
    Z = z * s

    wX = w * X; wY = w * Y; wZ = w * Z
    xX = x * X; xY = x * Y; xZ = x * Z
    yY = y * Y; yZ = y * Z; zZ = z * Z

    # not pretty...
    r00 = tf.where(Nq_is_zero, 1.0, 1.0 - (yY + zZ))
    r01 = tf.where(Nq_is_zero, 0.0, xY - wZ)
    r02 = tf.where(Nq_is_zero, 0.0, xZ + wY)
    r10 = tf.where(Nq_is_zero, 0.0, xY + wZ)
    r11 = tf.where(Nq_is_zero, 1.0, 1.0 - (xX + zZ))
    r12 = tf.where(Nq_is_zero, 0.0, yZ - wX)
    r20 = tf.where(Nq_is_zero, 0.0, xZ - wY)
    r21 = tf.where(Nq_is_zero, 0.0, yZ + wX)
    r22 = tf.where(Nq_is_zero, 1.0, 1.0 - (xX + yY))

    r0 = tf.stack([r00, r01, r02])
    r1 = tf.stack([r10, r11, r12])
    r2 = tf.stack([r20, r21, r22])

    R = tf.stack([r0, r1, r2])
    R = tf.transpose(R, (2, 0, 1))
    return R


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
    q0_is_positive = tf.greater_equal(q0, 0.)
    qi = tf.where(q0_is_positive, qi, -qi)
    qj = tf.where(q0_is_positive, qj, -qj)
    qk = tf.where(q0_is_positive, qk, -qk)
    q0 = tf.where(q0_is_positive, q0, -q0)

    Q = tf.transpose(tf.stack([q0, qi, qj, qk]))

    return Q
