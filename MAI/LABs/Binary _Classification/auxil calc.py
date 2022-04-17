import statistics as st
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

E1 = [[22, 147, 38], [34, 81, 15], [61, 43, 65], [21, 74, 45], [16, 59, 8],
      [52, 15, 29]]
Eav1 = [[round(st.mean(el), 2),round(st.mean(el)/900*100, 1)] for el in E1]
print(Eav1)

E2 = [[277, 168, 250], [143, 227, 299], [156, 166, 120], [141, 150, 133],
      [104, 83, 69], [76, 90, 100]]
Eav2 = [[round(st.mean(el), 2),round(st.mean(el)/900*100, 1)] for el in E2]
print(Eav2)
