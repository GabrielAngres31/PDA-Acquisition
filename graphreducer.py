# graphreducer

import numpy as np

def orientation(x1, y1, x2, y2, x3, y3):
    val = ((y2 - y1) * (x3 - x2)) - ((x2 - x1) * (y3 - y2))
    return np.sign(val)

def crossing(x1, y1, x2, y2, x3, y3, x4, y4):
    return orientation(x1, y1, x2, y2, x3, y3) * orientation(x1, y1, x2, y2, x4, y4)


print(crossing(0, 0, 2, 2, -1, -1, 0, 2))
print(crossing(0, 0, 2, 2, 0, 2, 2, 0))