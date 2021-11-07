from sklearn.metrics import confusion_matrix

y_true = [0,0,1,0]
y_pred = [0,0,0,0]
a=confusion_matrix(y_true, y_pred)
print(a)

#a=multilabel_confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))