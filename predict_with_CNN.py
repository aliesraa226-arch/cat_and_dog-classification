import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1️⃣ Data augmentation & loading dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "C:\\Users\\Administrator\\Desktop\\tasks_visual\\debi\\catanddog\\cat_and_dog\\training_set\\training_set",
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    "C:\\Users\\Administrator\\Desktop\\tasks_visual\\debi\\catanddog\\cat_and_dog\\test_set\\test_set", 
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

# 2️⃣ Building the CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 3️⃣ Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4️⃣ Train the model
history = model.fit(train_data, epochs=10, validation_data=test_data)

# 5️⃣ Evaluate
loss, acc = model.evaluate(test_data)
print("Test Accuracy:", acc)