import numpy as np
import cv2
import os
import random
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

i = 0
training_data = []
main = os.getcwd() + "/Training Images"
Categories = ['Five', 'Four', 'One', 'Three', 'Two', 'Zero']


def create_training():
    for category in Categories:
        path = os.path.join(main, category)
        class_num = Categories.index(category)
        for img in os.listdir(path):
            try:
                read_image = Image.open(os.path.join(path, img)).convert('L')
                read_image = np.array(read_image)
                read_image = read_image.reshape(1, 28, 28)
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                training_data.append([read_image, class_num])
            except Exception as e:
                pass


create_training()
random.shuffle(training_data)

validate_number = 1000
x_train = np.zeros((len(training_data)-validate_number, 28, 28))
y_train = np.zeros((len(training_data)-validate_number, 1))
y_train = y_train.flatten()

x_validate = np.zeros((validate_number, 28, 28))
y_validate = np.zeros(validate_number)
y_validate = y_validate.flatten()

k = 0
for x_data, y_data in training_data:
    if i < (len(training_data)-validate_number):
        x_train[i] = np.array(x_data)
        y_train[i] = np.array(y_data)
    else:
        x_validate[k] = np.array(x_data)
        y_validate[k] = np.array(y_data)
        k += 1
    i = i+1

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_validate = tf.keras.utils.normalize(x_validate, axis=1)

print(x_train.shape)
print(y_train.shape)

print(x_validate.shape)
print(y_validate.shape)

# --------------------------------------------------------------------------
# Training Begins

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
# Probability distribution, we want to use SOFT-MAX
model.add(tf.keras.layers.Dense(6, activation=tf.nn.softmax))

# Optimizer -> we can use adam, most default
# Loss -> sparse_categorical_crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
ep = 100
history = model.fit(x_train, y_train, epochs=ep, batch_size=128, validation_data=(x_validate, y_validate), verbose=2)

loss = history.history['loss']
val_losst = history.history['val_loss']
acc = history.history['acc']
val_acct = history.history['val_acc']

epochs = range(1, ep+1)

plt.plot(epochs, loss, 'ko', label='Training Loss')
plt.plot(epochs, val_losst, 'k', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'yo', label='Training Accuracy')
plt.plot(epochs, val_acct, 'y', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


val_loss, val_acc = model.evaluate(x_validate, y_validate)
# print(val_loss, val_acc)

model.save("gesture_final_model.model")


