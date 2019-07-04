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

    def test_euler2mat(self):
        batchsize = 128
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

        ax, ay, az = ELR.mat2euler(R_tf)
        euler_tf = tf.transpose(tf.stack([ax, ay, az]))

        euler_np = []
        for i in range(batchsize):
            ax, ay, az = transforms3d.euler.mat2euler(R_np[i])
            euler_np.append((ax, ay, az))
        euler_np = np.array(euler_np, dtype=np.float32)

        Dif = np.abs(euler_np - euler_tf)
        self.assertAlmostEqual(0., np.max(Dif), places=5)


if __name__ == '__main__':
    unittest.main()
