import numpy as np
from collections import namedtuple

TIME_FACTOR = 0
NOISE_BIAS = 0
OUTPUT_NOISE = [False] * 30
OUTPUT_SIZE = 30







def activations(a):
  a = np.tanh(a)
  a[1] = (a[1] + 1) / 2
  a[2] = (a[2] + 1) / 2
  return a

class Controller():
    def __init__(self):
        self.time_factor = TIME_FACTOR
        self.noise_bias = NOISE_BIAS
        self.output_noise=OUTPUT_NOISE
        self.activations=activations
        self.output_size = OUTPUT_SIZE

        


