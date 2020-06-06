import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import preprocess_crop

import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


#DEFAULT MODEL PARAMETERS
batch_size = 128
epochs = 10
IMG_HEIGHT = 150
IMG_WIDTH = 150
train_dir = r"/home/bloke/Documents/Birdshit/smart-bird-house/data/model2_data"

validation_split = .1



train_pos_dir = os.path.join(train_dir, 'lila_dataset_bird')
train_neg_dir = os.path.join(train_dir, 'lila_dataset_squirrel')

print(len(os.listdir(train_pos_dir)), " number of positive training instances")
print(len(os.listdir(train_neg_dir)), " number of negative training instances")
#print(len(os.listdir(val_pos_dir)), " number of positive validation instances")
#print(len(os.listdir(val_neg_dir)), " number of negative validation instances")

total_train = (len(os.listdir(train_pos_dir)) + len(os.listdir(train_neg_dir)))*(1 - validation_split)

total_val = (len(os.listdir(train_pos_dir)) + len(os.listdir(train_neg_dir)))*validation_split

#total_val = len(os.listdir(val_pos_dir)) + len(os.listdir(val_neg_dir))

train_image_generator = ImageDataGenerator(validation_split=.1, rescale=1./255) # Generator for our training data
#validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           subset='training',
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           interpolation='lanczos:center',
                                                           class_mode='binary')

val_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                            subset='validation',
                                                              directory=train_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


sample_training_images, _ = next(train_data_gen)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Might need to change model metrics


history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    #validation_split=validation_split,
    #shuffle=True
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
   
)


model.save('model2_save')


#Plotting Parameters
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

#Plotting accuracy and loss over epochs
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
