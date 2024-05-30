import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Set the path for the dataset
DATA_PATH = r'C:\\Users\\tekul\\OneDrive\\Desktop\\CodeAlpha_mltasks2\\emotion_dataset\\wav'

# Function to extract features from audio files
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, Error: {e}")
        return None

# Load data and extract features
def load_data(data_path):
    features = []
    labels = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        if file_path.endswith(".wav"):
            label = file[0:2]  # Assume the first two characters of the filename represent the emotion
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)

# Prepare the dataset
X, y = load_data(DATA_PATH)

# Verify that data was loaded correctly
if X.size == 0 or y.size == 0:
    raise ValueError("No valid data found. Please check the dataset path and the audio files.")

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)


# Build the model
model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_categorical.shape[1], activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Save the model
model.save('emotion_recognition_model.keras')

# Save the label encoder
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


