import requests
from PIL import Image
from io import BytesIO
import os
import numpy as np
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
  "training/",
  validation_split=0.2, # Untuk Mmevalidasi 20% dan 80% untuk membuat pelatihan/model
  subset="training",
  seed=123, # Membuat dataset Konsisten
  image_size=(350, 350), #(tinggi,lebar) gambar
  batch_size=32) # Dataset Disimpan Sebanyak 32

valid_ds = tf.keras.utils.image_dataset_from_directory(
  "training/",
  validation_split=0.2, # Untuk Mmevalidasi 20% dan 80% untuk membuat pelatihan/model
  subset="training",
  seed=123, # Membuat dataset Konsisten
  image_size=(350, 350), #(tinggi,lebar) gambar
  batch_size=32) # Dataset Disimpan Sebanyak 32

print(train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(350, 350, 3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(350, 350, 3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=(350, 350, 3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


epochs=15
history = model.fit(
  train_ds,
  validation_data=valid_ds,
  epochs=epochs
)

# img = Image.open("./rab.jpg")
# resize = img.resize((350,350))

# img_array = tf.keras.utils.img_to_array(resize)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
# pred = model.predict(img_array)
# print(np.argmax(pred))


print("Example Link = https://contoh.com/image.jpeg")
url = str(input("Image Link = "))

res = requests.get(url)

if res.status_code == 200:
	img = Image.open(BytesIO(res.content))
	resize = img.resize((350,350))
	img_array = tf.keras.utils.img_to_array(resize)
	img_array = tf.expand_dims(img_array, 0) 
	pred = model.predict(img_array)
	score = np.argmax(pred)

	if score == 0:
		print("Is A Cats")
	else:
		print("Is A Rabbits")
