import numpy as np
import random

#1 patterns
pattern1 = [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0]]
pattern2 = [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
                 [0, 1, 1, 1, 0]]
pattern3 = [[0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0]]
pattern4 = [[0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0],
                 [0, 1, 0, 0, 0]]
pattern5 = [[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0],
                 [1, 0, 0, 0, 0]]
pattern6 = [[0, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 0]]

#0 patterns
pattern7 = [[0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0],
                 [0, 1, 1, 1, 0]]
pattern8 = [[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0],
                 [0, 0, 1, 0, 0]]
pattern9 = [[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0],
                 [0, 1, 1, 1, 0]]
pattern10 = [[0, 0, 1, 1, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1], [0, 1, 0, 1, 0],
                 [0, 1, 1, 0, 0]]
pattern11 = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1]]
pattern12 = [[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1],
                   [0, 1, 1, 1, 0]]

Data = []

for n in range(1, 13):
    Data.extend(eval('pattern' + str(n)))

Data = np.reshape(Data, (12, 25))
print(Data)

Desired_output = [1]*6 + [-1]*6

#import Matrix_ops