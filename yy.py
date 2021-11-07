import matplotlib.pyplot as plt
fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')
acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')
plt.show()

import numpy as np
import math


a = np.arange(9) -4
# b = a.reshape(3)
# print(b)
print(a)
v = np.linalg.norm(a)
print(v)