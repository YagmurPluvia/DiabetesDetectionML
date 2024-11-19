import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, cohen_kappa_score, f1_score, roc_curve, auc, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)#değiştirilebilir----->test_size=0.2?

#Model oluşturma
# Define the Keras Sequential model
model = Sequential()

#Modele katman ekleme
model.add(Dense(units=128, activation='relu', input_dim=X.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Dropout for regularization

model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback'i oluşturma ---------------------------------->>
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit the model ------->epoch değiştirilebilri
history = model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping])

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

print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(Y_test, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ANN ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Performans Metrikleri
print("Classification Report:\n", classification_report(Y_test, predictions))

accuracy = accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy}')

sensitivity = recall_score(Y_test, predictions)
print(f'Sensitivity (Recall): {sensitivity}')

specificity = recall_score(Y_test, predictions, pos_label=0)
print(f'Specificity: {specificity}')

kappa = cohen_kappa_score(Y_test, predictions)
print(f'Kappa Katsayısı: {kappa}')

f1 = f1_score(Y_test, predictions)
print(f'F-ölçümü: {f1}')

roc_auc = roc_auc_score(Y_test, predictions)
print(f'AUC Katsayısı: {roc_auc}')



