import numpy as np

with open('imgs/loclearn/0_500.jpg', 'r') as f:
    a = np.fromfile(f, dtype=np.int32)
with open('imgs/loclearn/0_3000.jpg', 'r') as f:
    b = np.fromfile(f, dtype=np.int32)
print(a)
print(b)
print(np.mean(a - b[:len(a)]))
print(np.var(a - b[:len(a)]))