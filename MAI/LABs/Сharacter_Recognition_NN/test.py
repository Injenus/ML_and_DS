import numpy as np
import torch
a = np.zeros((3, 3))

b = np.array([[0.53636, 0.091212], [0.00004, 0.000005]])

print(np.ceil(np.around(b, 1)))

c = np.array([[1.1, 2], [4, 5]])
print(c*2)

d = np.multiply(b, np.eye(2))

print(np.around(d))
print(np.eye(3))

print(d + np.eye(2))

print(ord("H"))

print(chr(73))

sc = 72
x = [chr(i) for i in range(sc, sc + 10)]
print(x)


inp = torch.from_numpy(d)  # входы для Nsymbols букв
print(torch.cat((inp,inp),0))


y=[1,8,9,9,-3,10,3,8]

print(max(c))

# plt.clf()
# plt.subplot(2, 2, 1)
# plt.imshow(testing(our_network, inp), cmap='RdPu', clim=(-1, 1))
# plt.title('Идеал. НН по идеал. данным')
# plt.subplot(2, 2, 2)
# plt.imshow(testing(our_network, inp_noise_train), cmap='RdPu', clim=(-1, 1))
# plt.title('Идеал. НН по шум. данным')
# plt.subplot(2, 2, 3)
# plt.imshow(testing(networkByNoise, inp), cmap='RdPu', clim=(-1, 1))
# plt.title('Шум. НН по идеал. данным')
# plt.subplot(2, 2, 4)
# plt.imshow(testing(networkByNoise, inp_noise_train), cmap='RdPu', clim=(-1, 1))
# plt.title('Шум. НН по шум. данным')
# plt.subplots_adjust(wspace=0.25, hspace=0.3)
# plt.savefig('TOTAL_conf_matrxs_TRAIN_noise.jpg')

# testing(our_network, inp, 'Идеальная_по_идеальным')
# testing(networkByNoise, inp_noise_test, 'Шумовая_по_test_зашумленным')
# testing(our_network, inp_noise_test, 'Идеальная_по_test_зашумленным')
# testing(our_network, inp_noise_train, 'Идеальная_по_train_зашумленным')
# testing(networkByNoise, inp_noise_train, 'Шумовая_по_train_зашумленным')

# plt.clf()
# plt.subplot(1, 2, 1)
# plt.imshow(testing(our_network, inp_noise_test), cmap='RdPu', clim=(-1, 1))
# plt.title('Идеал. НН по шум. тест. данным')
# plt.subplot(1, 2, 2)
# plt.imshow(testing(our_network, inp_noise_train), cmap='RdPu', clim=(-1, 1))
# plt.title('Идеал. НН по шум. трэин. данным')
# plt.subplots_adjust(wspace=0.25, hspace=0.3)
# plt.savefig('test_train.jpg')

# confMatr_ideal_for_ideal = confMatr_ideal_for_ideal + np.around(
#     np.multiply(testing(our_network, inp), np.eye(Nsymbols)), 0)
#
# confMatr_noise_for_ideal = confMatr_noise_for_ideal + np.around(
#     np.multiply(testing(our_network, inp_noise_test), np.eye(Nsymbols)), 0)
#
# confMatr_ideal_for_noise = confMatr_ideal_for_noise + np.around(
#     np.multiply(testing(networkByNoise, inp), np.eye(Nsymbols)), 0)
#
# confMatr_noise_for_noise = confMatr_noise_for_noise + np.around(
#     np.multiply(testing(networkByNoise, inp_noise_test), np.eye(Nsymbols)), 0)

# confMatr_ideal_for_ideal = confMatr_ideal_for_ideal + np.ceil(
#     np.around(np.multiply(testing(our_network, inp), np.eye(Nsymbols)), 1))
#
# confMatr_noise_for_ideal = confMatr_noise_for_ideal + np.ceil(np.around(
#     np.multiply(testing(our_network, inp_noise_test), np.eye(Nsymbols)), 1))
#
# confMatr_ideal_for_noise = confMatr_ideal_for_noise + np.ceil(
#     np.around(np.multiply(testing(networkByNoise, inp), np.eye(Nsymbols)), 1))
#
# confMatr_noise_for_noise = confMatr_noise_for_noise + np.ceil(np.around(
#     np.multiply(testing(networkByNoise, inp_noise_test), np.eye(Nsymbols)), 1))

# confMatr_ideal_for_ideal = confMatr_ideal_for_ideal + np.multiply(
#     testing(our_network, inp), np.eye(Nsymbols))
#
# confMatr_noise_for_ideal = confMatr_noise_for_ideal + np.multiply(
#     testing(our_network, inp_noise_test), np.eye(Nsymbols))
#
# confMatr_ideal_for_noise = confMatr_ideal_for_noise + np.multiply(
#     testing(networkByNoise, inp), np.eye(Nsymbols))
#
# confMatr_noise_for_noise = confMatr_noise_for_noise + np.multiply(
#     testing(networkByNoise, inp_noise_test), np.eye(Nsymbols))