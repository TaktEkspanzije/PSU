import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()


x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255


x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)


y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


model.summary()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    x_train_s, y_train_s,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

train_loss, train_acc = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Tocnost na train skupu: {train_acc*100:.2f}%")
print(f"Tocnost na test skupu: {test_acc*100:.2f}%")

y_pred = model.predict(x_test_s)
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()


incorrect_idx = np.where(y_pred_labels != y_test)[0]
plt.figure(figsize=(10, 4))
for i, idx in enumerate(incorrect_idx[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Prava: {y_test[idx]}, Pred: {y_pred_labels[idx]}")
    plt.axis('off')
plt.show()