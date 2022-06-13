import numpy as np
bgr = np.array([.47,.1,.6]) #bgr为背景
bgr = np.resize(bgr,[3,1,1])
print(bgr)
cgr = np.array([[[0.47]],[[0.1 ]],[[0.6 ]]])
print(cgr)