import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt

# Load and preprocess data
veriler = pd.read_csv('veriler_isleme.csv')
eksik_veri = veriler.iloc[:, 1:6]
df = pd.DataFrame(eksik_veri)
df.replace(0, np.nan, inplace=True)
ortalama = df.mean()
for column in df.columns:
    df[column].fillna(ortalama[column], inplace=True)

doğ = veriler.iloc[:, :1]
sonveriler = pd.concat([doğ, df], axis=1)
sonveriler = pd.concat([sonveriler, veriler.iloc[:, -2:]], axis=1)

X = sonveriler.iloc[:, 0:8].values
Y = sonveriler.iloc[:, -1].values  # Assuming Y is binary class

# Standardization
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the Keras Sequential model
model = Sequential()

# Add layers to the model
model.add(Dense(units=64, activation='relu', input_dim=X.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Dropout for regularization

model.add(Dense(units=32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

# Plot training history: Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot training history: Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Predictions
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)  # Thresholding probabilities to get binary predictions
accuracy = accuracy_score(Y_test, predictions)
conf_matrix = confusion_matrix(Y_test, predictions)

print("Test Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)

import seaborn as sns
from sklearn.metrics import classification_report, cohen_kappa_score

# Calculate additional metrics
tn, fp, fn, tp = conf_matrix.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
f1_score = 2 * (precision * recall) / (precision + recall)
kappa = cohen_kappa_score(Y_test, predictions)

print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Specificity:", specificity)
print("F1 Score:", f1_score)
print("Kappa Score:", kappa)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
