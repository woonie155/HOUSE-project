import numpy as np

gesture = ['ALT_TAB', 'ALT_F4', 'FULL', 'SOUND_CONTROL']

data = np.concatenate([
    np.load('dataset/seq_ALT_TAB.npy'),
    np.load('dataset/seq_ALT_F4.npy'),
    np.load('dataset/seq_FULL.npy'),
    np.load('dataset/seq_SOUND_CONTROL.npy'),
], axis=0)

x_data = data[:, :, :-1] 
labels = data[:, 0, -1]

from tensorflow.keras.utils import to_categorical
y_data = to_categorical(labels, num_classes=len(gesture))

from sklearn.model_selection import train_test_split
x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=True, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=1)

print("최종")
print(x_train.shape, y_train.shape) 
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(gesture), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100,
    callbacks=[
        ModelCheckpoint('models/cursor_model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=20, verbose=1, mode='auto')
    ]
)

from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.models import load_model
model = load_model('models/cursor_model.h5')
y_pred = model.predict(x_test)
test_p = multilabel_confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

for i in range(len(test_p)):
    print(gesture[i],"\n", test_p[i], "\n")
    
