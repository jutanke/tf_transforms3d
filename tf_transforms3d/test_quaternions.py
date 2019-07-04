import sys
sys.path.insert(0, './../')
import unittest
import transforms3d
import numpy as np
import numpy.random as rnd
import tensorflow as tf
import math as m
import tf_transforms3d.quaternions as QTS

class TestQuaternions(unittest.TestCase):

    def test_qmult(self):
        batchsize = 1024
        Q1 = self.get_quat(self.get_random_mat(batchsize=batchsize))
        Q2 = self.get_quat(self.get_random_mat(batchsize=batchsize))

        prod_np = []
        for i in range(batchsize):
            q = transforms3d.quaternions.qmult(Q1[i], Q2[i])
            prod_np.append(q)
        prod_np = np.array(prod_np, dtype=np.float32)

        prod_tf = QTS.qmult(Q1, Q2)
        self.assertAlmostEqual(0., self.maxdif(prod_np, prod_tf), places=5)


    def test_quat2mat(self):
        batchsize = 1024
        R = self.get_random_mat(batchsize=batchsize)
        Q_np = self.get_quat(R)
        R_np = self.quat2mat(Q_np)

        self.assertAlmostEqual(0., self.maxdif(R, R_np), places=5)

        R_tf = QTS.quat2mat(Q_np)
        self.assertAlmostEqual(0., self.maxdif(R, R_tf), places=5)


    def test_mat2quat(self):

        batchsize = 1024
        R_np = self.get_random_mat(batchsize=batchsize)
        Q_np = self.get_quat(R_np)

        Q_tf = QTS.mat2quat(R_np)

        Dif = np.abs(Q_np - Q_tf)

        MAX_LOC = np.argmax(Dif) // 4

        self.assertAlmostEqual(0., self.maxdif(Q_np, Q_tf), places=4)

    # ==============================================

    def maxdif(self, A, B):
        Dif = np.abs(A - B)
        return np.max(Dif)
    
    def get_quat(self, R):
        batchsize = len(R)
        Q_np = []
        for i in range(batchsize):
            q = transforms3d.quaternions.mat2quat(R[i])
            Q_np.append(q)
        Q_np = np.array(Q_np)
        return Q_np.astype('float32')

    def quat2mat(self, Q):
        batchsize = len(Q)
        R_np = []
        for i in range(batchsize):
            R = transforms3d.quaternions.quat2mat(Q[i])
            R_np.append(R)
        R_np = np.array(R_np)
        return R_np.astype('float32')

    def get_random_mat(self, batchsize=1024):
        euler = (rnd.random(size=(batchsize, 3)) - 0.5) * 2 * m.pi
        euler = euler.astype('float32')
        R_np = []
        for i in range(batchsize):
            R = transforms3d.euler.euler2mat(*euler[i], axes='sxyz')
            R_np.append(R)
        R_np = np.array(R_np)
        return R_np.astype('float32')

if __name__ == '__main__':
    unittest.main()
