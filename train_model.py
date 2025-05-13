import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load metadata
df = pd.read_csv('../dataset/HAM10000_metadata.csv')

# Use only a subset of classes for simplicity
df = df[df['dx'].isin(['mel', 'nv', 'bkl'])]  # melanoma, nevus, keratosis

# Map labels to numbers
label_map = {'mel': 0, 'nv': 1, 'bkl': 2}
df['label'] = df['dx'].map(label_map)

# Load and preprocess images
image_dir1 = '../dataset/HAM10000_images_part_1'
image_dir2 = '../dataset/HAM10000_images_part_2'

X, y = [], []

for index, row in df.sample(1000, random_state=42).iterrows():  # Only 1000 random images

    image_id = row['image_id']
    label = row['label']
    
    image_path = os.path.join(image_dir1, image_id + '.jpg')
    if not os.path.exists(image_path):
        image_path = os.path.join(image_dir2, image_id + '.jpg')

    if os.path.exists(image_path):
        img = Image.open(image_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        X.append(img_array)
        y.append(label)

X = np.array(X)
y = to_categorical(np.array(y), num_classes=3)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), batch_size=32)

# Save model
model.save('../skin_disease_model.h5')
