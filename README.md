# tf_transforms3d
3 dimensional spatial transformations for tensorflow 2.0, heavily inspired by [transforms3d](https://github.com/matthew-brett/transforms3d).

## Install
```
pip install git+https://github.com/jutanke/tf_transforms3d.git
```

## Usage
This library only supports one rotational setup (xyz) at the moment!

All functions within this library expect batch-inputs, e.g.:

```python
import numpy as np
import numpy.random as rnd
import math as m
import tf_transforms3d.euler as EULER
import tf_transforms3d.quaternions as QUAT


batchsize = 1024
euler = (rnd.random(size=(batchsize, 3)) - 0.5) * 2 * m.pi  # random euler angles

quat = EULER.euler2quat(euler)  # --> (batchsize x 4)
mat = EULER.euler2mat(euler)  # --> (batchsize x 3 x 3)

euler_ = EULER.mat2euler(mat)  # --> (batchsize x 3)
euler_ = EULER.quat2mat(quat)  # --> (batchsize x 3)

mat_ = QUAT.quat2mat(quat)  # --> (batchsize x 3 x 3)
quat_ = QUAT.mat2quat(mat)  # --> (batchsize x 4)

q_prod = QUAT.qmult(quat_, quat)  # --> (batchsize x 4)
```
