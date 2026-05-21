import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

img_size = (48, 48)
model = tf.keras.models.load_model('best_model.h5')

img_path = 'Zebra.jpg'
img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
print(f"Predviđena klasa: {predicted_class}")

plt.imshow(image.load_img(img_path, target_size=img_size))
plt.title(f"Predviđena klasa: {predicted_class}")
plt.axis('off')
plt.show()