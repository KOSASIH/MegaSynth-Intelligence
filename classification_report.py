import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

# Load and preprocess the dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/train/dataset',
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/train/dataset',
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# Create the deep learning model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

# Evaluate the model on the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/test/dataset',
    labels='inferred',
    label_mode='categorical',
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

loss, accuracy = model.evaluate(test_dataset)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Save the trained model
model.save('path/to/save/model')

# Generate performance metrics on the test dataset
test_predictions = model.predict(test_dataset)
test_labels = tf.concat([y for x, y in test_dataset], axis=0)
test_labels = tf.argmax(test_labels, axis=1)
test_predictions = tf.argmax(test_predictions, axis=1)

print(classification_report(test_labels, test_predictions))
