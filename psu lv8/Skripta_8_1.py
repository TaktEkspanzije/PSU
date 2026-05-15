from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# TODO: strukturiraj konvolucijsku neuronsku mrezu

model = models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', padding='valid', input_shape=(28, 28, 1)),
layers.MaxPooling2D(pool_size=(2, 2), strides=2),
layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
layers.MaxPooling2D(pool_size=(2, 2), strides=2),
layers.Flatten(),
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')
])
model.summary()


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# TODO: definiraj callbacks
my_callbacks = [keras.callbacks.TensorBoard(log_dir='logs', update_freq=100), 
                keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True)
]


# TODO: provedi treniranje mreze pomocu .fit()

model.fit(x_train_s, y_train_s, epochs=10, batch_size=64, validation_split=0.1, callbacks=my_callbacks)

#TODO: Ucitaj najbolji model

best_model = models.load_model('best_model.keras')

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje

y_train_pred = np.argmax(best_model.predict(x_train_s), axis=1)
y_test_pred  = np.argmax(best_model.predict(x_test_s),  axis=1)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy  = accuracy_score(y_test,  y_test_pred)

print(f"Tocnost na skupu za ucenje:   {train_accuracy * 100:.2f}%")
print(f"Tocnost na skupu za testiranje: {test_accuracy * 100:.2f}%")

# TODO: Prikazite matricu zabune na skupu podataka za testiranje

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test  = confusion_matrix(y_test,  y_test_pred)

print("\nMatrica zabune - skup za ucenje:")
print(cm_train)

print("\nMatrica zabune - skup za testiranje:")
print(cm_test)