from LegendaryNinja_v0 import LegendaryNinja_v0
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

env = LegendaryNinja_v0()

myarray = env.step(0)[0]


dummy = cv2.resize(myarray, (600, 600))

x_t1 = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY)

print(np.shape(x_t1))

x_t1 = np.reshape(x_t1, (600, 600, 1))

print(x_t1)

##s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

print(type(x_t1))