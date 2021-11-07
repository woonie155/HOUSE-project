import numpy as np

actions = ['ALT_TAB', 'ALT_F4', 'ENTER', 'SOUND_CONTROL']

data = np.concatenate([
    np.load('dataset/seq_ALT_TAB.npy'),
    np.load('dataset/seq_ALT_F4.npy'),
    np.load('dataset/seq_ENTER.npy'),
    np.load('dataset/seq_SOUND_CONTROL.npy'),
], axis=0)

X = data[:, :, :-1] 
labels = data[:, 0, -1]

from tensorflow.keras.utils import to_categorical
Y = to_categorical(labels, num_classes=len(actions))

from sklearn.model_selection import train_test_split
X = X.astype(np.float32)
Y = Y.astype(np.float32)

x_data = X[:2800]
y_data = Y[:2800]
print(x_data)
print(y_data)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

print(x_train.shape, y_train.shape) 
print(x_val.shape, y_val.shape) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100,
    callbacks=[
        ModelCheckpoint('models/cursor_model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)





# import matplotlib.pyplot as plt
# fig, loss_ax = plt.subplots(figsize=(16, 10))
# acc_ax = loss_ax.twinx()
# loss_ax.plot(history.history['loss'], 'y', label='train loss')
# loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# loss_ax.legend(loc='upper left')
# acc_ax.plot(history.history['acc'], 'b', label='train acc')
# acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
# acc_ax.set_ylabel('accuracy')
# acc_ax.legend(loc='upper left')
# plt.show()

# from sklearn.metrics import multilabel_confusion_matrix
# from tensorflow.keras.models import load_model
# model = load_model('models/cursor_model.h5')
# y_pred = model.predict(x_val)
# print(y_pred)
# multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))