# Library Import
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Custom helper import
from helper.cadplot import conf_matrix
from helper.utils import log_results, plot_training_history,evaluate_model

# Trial run for mini dataset due to processing times.
trail_run = True


# Path Define
# File prefix name
_prefix = ''
_datapath="/Users/taiaburrahman/Desktop/Udg/CADx/Challenge/tf/"
# Set the path to your datasets
if trail_run:
    path = _datapath + "trial_dataset/multiCLASS/"
    _prefix = "trail_"
else:
    path = _datapath + "dataset/multiCLASS/"
    _prefix = "final_"

TRAIN_DATA_DIR = path + 'train'
VALIDATION_DATA_DIR = path + 'val'
TEST_DATA_DIR = path + 'testX'


# Set a random seed for reproducibility
seed_value = 42
tf.random.set_seed(seed_value)

# Helper function

# Save model 
def model_save(model,file_path):
    model.save(file_path)

    # Model Param setup
# Image dimensions
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
# Batch size
batch_size = 32
# Iteration
EPOCHS = 20


# Use ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=90,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2  # 20% of the data will be used for validation
)
# Use ImageDataGenerator for testing without data augmentation
test_datagen = ImageDataGenerator(rescale=1.0/255)
# Create a data generator for training (with validation split)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical',
    subset='training'  # specify that this is the training set
)

# Create a data generator for validation (with validation split)
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical',
    subset='validation'  # specify that this is the validation set
)

test_generator = test_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

prefix = _prefix + 'mm_VGG16'
# Load the pre-trained VGG16 model
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the convolutional layers of VGG16
for layer in vgg16_model.layers:
    layer.trainable = False

# Create a new model with VGG16 and additional layers for binary classification
vgg16_custom_model = Sequential()
vgg16_custom_model.add(vgg16_model)
vgg16_custom_model.add(Flatten())
vgg16_custom_model.add(Dense(128, activation='relu'))
vgg16_custom_model.add(Dense(1, activation='sigmoid'))

# Model summary
vgg16_custom_model.summary()

# Compile the VGG16 custom model
vgg16_custom_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the VGG16 custom model
start_time = time.time()

history_vgg16_custom = vgg16_custom_model.fit(
        train_generator, 
        epochs=EPOCHS, 
        validation_data=validation_generator,
    )
end_time = time.time()

# Evaluate the VGG16 custom model on the test set
test_loss_vgg16_custom, test_accuracy_vgg16_custom = vgg16_custom_model.evaluate(test_generator)

# Save the VGG16 custom model (modify the saving path)
model_save(vgg16_custom_model, f'model/{prefix}_model.h5')


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np

# Generate predictions on the test set
y_pred = vgg16_custom_model.predict(test_generator)

# Convert one-hot encoded predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert one-hot encoded true labels to class labels
y_true_labels = test_generator.classes

# Calculate evaluation metrics
accuracy = accuracy_score(y_true_labels, y_pred_labels)
precision = precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=1)
recall = recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=1)
f1 = f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=1)
kappa = cohen_kappa_score(y_true_labels, y_pred_labels)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Cohen\'s Kappa: {kappa:.4f}')

# Print class-wise metrics
print(classification_report(y_true_labels, y_pred_labels))

custom_model = vgg16_custom_model
model_history = history_vgg16_custom
# Assuming you have variables vgg16_custom_model and test_generator
test_loss, test_accuracy, conf_mat, classification_rep = evaluate_model(custom_model, test_generator)
# Assuming you have variables like start_time, end_time, test_accuracy, conf_mat, and classification_rep
log_results(prefix, start_time, end_time, test_accuracy, conf_mat, classification_rep)
print(model_history.history.keys())
# Assuming you have a variable model and prefix
plot_training_history(model_history, prefix)
# Create a heatmap for the confusion matrix
conf_matrix(conf_mat)