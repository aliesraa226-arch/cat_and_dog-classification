
# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest',
                                   validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    "C:\\Users\\Administrator\\Desktop\\tasks_visual\\debi\\catanddog\\cat_and_dog\\training_set\\training_set",
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    "C:\\Users\\Administrator\\Desktop\\tasks_visual\\debi\\catanddog\\cat_and_dog\\training_set\\training_set",
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

model = Sequential([
    Flatten(input_shape=(128,128,3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#callback
early_stop=EarlyStopping(monitor='val_loss',patience=5)
reduce_lr=ReduceLROnPlateau(monitor='val_loss',patience=3,verbose=1)
checkpoint=ModelCheckpoint('best_model.h5',monitor='val_loss',save_best_only=True,verbose=1)

model.fit(train_data, validation_data=val_data, epochs=10 ,callbacks=[early_stop,reduce_lr,checkpoint])

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    "C:\\Users\\Administrator\\Desktop\\tasks_visual\\debi\\catanddog\\cat_and_dog\\test_set\\test_set",
    target_size=(128,128),
    batch_size=16,
    class_mode='binary'
)

loss, acc = model.evaluate(test_data)
print("Accuracy:", acc)

#pretrained model
base_model = MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet'
)
pretrained_model=Sequential([
    base_model,
    Flatten(),
    Dense(1, activation='sigmoid'),

])

pretrained_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_pretrained=ModelCheckpoint('best_pretrainedmodel.h5',monitor='val_accuracy',save_best_only=True,verbose=1)
pretrained_model.fit(train_data, validation_data=val_data, epochs=20 ,callbacks=[early_stop,reduce_lr,checkpoint_pretrained])

y_pred = pretrained_model.predict(test_data)
y_pred_classes = (y_pred > 0.5).astype(int)
y_true = test_data.classes
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
disp.plot(cmap=plt.cm.Blues)
plt.show()


