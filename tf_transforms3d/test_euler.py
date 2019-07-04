import sys
sys.path.insert(0, './../')
import unittest
import transforms3d
import numpy as np
import numpy.random as rnd
import tensorflow as tf
import math as m
import tf_transforms3d.euler as ELR

class TestEuler(unittest.TestCase):

    def test_euler2quat(self):
        batchsize = 1024
        euler = (rnd.random(size=(batchsize, 3)) - 0.5) * 2 * m.pi
        euler = euler.astype('float32')
        Q_np = []
        for i in range(batchsize):
            Q = transforms3d.euler.euler2quat(*euler[i], axes='sxyz')
            Q_np.append(Q)
        Q_np = np.array(Q_np)
        Q_tf = ELR.euler2quat(euler)
        Dif = np.abs(Q_np - Q_tf)
        N = np.linalg.norm(Q_tf, axis=1)
        self.assertAlmostEqual(1., np.min(N), places=4)
        self.assertAlmostEqual(1., np.max(N), places=4)
        self.assertAlmostEqual(0., np.max(Dif), places=5)

    def test_euler2mat(self):
        batchsize = 1024
        euler = (rnd.random(size=(batchsize, 3)) - 0.5) * 2 * m.pi
        euler = euler.astype('float32')
        R_np = []
        for i in range(batchsize):
            R = transforms3d.euler.euler2mat(*euler[i], axes='sxyz')
            R_np.append(R)
        R_np = np.array(R_np)
        R_tf = ELR.euler2mat(euler)
        Dif = np.abs(R_np - R_tf)
        ELR.mat2euler(R_tf)
        self.assertAlmostEqual(0., np.max(Dif), places=5)
    
    def test_mat2euler(self):
        batchsize = 1024
        euler = (rnd.random(size=(batchsize, 3)) - 0.5) * 2 * m.pi
        euler = euler.astype('float32')
        R_np = []
        for i in range(batchsize):
            R = transforms3d.euler.euler2mat(*euler[i], axes='sxyz')
            R_np.append(R)
        R_np = np.array(R_np)
        R_tf = ELR.euler2mat(euler)
        Dif = np.abs(R_np - R_tf)
        ELR.mat2euler(R_tf)
        self.assertAlmostEqual(0., np.max(Dif), places=5)

        euler_tf = ELR.mat2euler(R_tf)

        euler_np = []
        for i in range(batchsize):
            ax, ay, az = transforms3d.euler.mat2euler(R_np[i])
            euler_np.append((ax, ay, az))
        euler_np = np.array(euler_np, dtype=np.float32)

        Dif = np.abs(euler_np - euler_tf)
        self.assertAlmostEqual(0., np.max(Dif), places=5)

    def test_quat2euler(self):
        batchsize = 1024
        euler = (rnd.random(size=(batchsize, 3)) - 0.5) * 2 * m.pi
        Q_np = []
        for i in range(batchsize):
            Q = transforms3d.euler.euler2quat(*euler[i], axes='sxyz')
            Q_np.append(Q)
        Q_np = np.array(Q_np, dtype=np.float32)

        euler_np = []
        for i in range(batchsize):
            eul = transforms3d.euler.quat2euler(Q_np[i])
            euler_np.append(eul)
        euler_np = np.array(euler_np, dtype=np.float32)

        euler_tf = ELR.quat2euler(Q_np)

        Dif = np.abs(euler_tf - euler_np)
        self.assertAlmostEqual(0., np.max(Dif), places=4)



if __name__ == '__main__':
    unittest.main()
