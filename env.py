import numpy as np
from chainer import cuda

import socket
xp = np
if socket.gethostname() == "naruto":
    xp = cuda.cupy
    print("gpu mode")
