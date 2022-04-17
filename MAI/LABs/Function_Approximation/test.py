import math
import matplotlib.pyplot as plt

NN = [3, 5, 7, 9, 12, 25, 50, 70]
MSE = [0.4361, 0.0479, 0.0064, 0.0092, 0.0115, 0.0134, 0.0148, 0.0129]
# MSE_log = []
# for i in range(len(MSE)):
#     MSE_log.append(math.log(MSE[i]))

plt.title('Зависимость MSE от NN')
plt.plot(NN, MSE)
plt.xlabel('NN')
plt.ylabel('MSE')
plt.grid()
plt.savefig("MSEonNN.png")

