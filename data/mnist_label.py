import numpy as np
from sklearn.datasets import fetch_mldata


def save_np(out_file_name, num):
    np.save(out_file_name + ".npy", num)
    np.savetxt(out_file_name + ".csv", num, delimiter=",")
    return


mnist = fetch_mldata('MNIST original', data_home=".")
data_max = 70000
tmp = np.zeros((10, 10000)).astype(np.int32)
count = np.zeros(10).astype(np.int32)

for i in range(data_max):
    label = int(mnist.target[i])
    tmp[label][count[label]] = i
    count[label] += 1

save_np("./mnist/label", tmp)
save_np("./mnist/count", count)

print("end")
