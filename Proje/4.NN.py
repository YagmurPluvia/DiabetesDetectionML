import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, cohen_kappa_score, f1_score, roc_auc_score
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
sonveriler = pd.concat([sonveriler, veriler.iloc[:, -3:]], axis=1)

X = sonveriler.iloc[:, 0:8].values
Y = sonveriler.iloc[:, -1].values  # Assuming Y is binary class

# Standardization
sc = StandardScaler()
X = sc.fit_transform(X)

# Define the Keras Sequential model
def create_model():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=X.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
conf_matrices = []
validation_losses = []
validation_accuracies = []

for train, test in cv.split(X, Y):
    model = create_model()
    history = model.fit(X[train], Y[train], epochs=50, batch_size=32, verbose=0)
    
    # Evaluation on validation set
    _, accuracy = model.evaluate(X[test], Y[test], verbose=0)
    accuracies.append(accuracy)
    
    # Validation loss and accuracy
    val_loss, val_acc = model.evaluate(X[test], Y[test], verbose=0)
    validation_losses.append(val_loss)
    validation_accuracies.append(val_acc)
    
    # Confusion matrix
    y_pred = model.predict_classes(X[test])
    conf_matrices.append(confusion_matrix(Y[test], y_pred))

# Display results
print("Accuracy for each fold:", accuracies)
print("Mean Accuracy:", np.mean(accuracies))
print("Confusion Matrices for each fold:", conf_matrices)
print("Validation Losses for each fold:", validation_losses)
print("Validation Accuracies for each fold:", validation_accuracies)

# Classification report and metrics (for last fold)
print("\nClassification Report:\n")
print(classification_report(Y[test], y_pred))

accuracy = accuracy_score(Y[test], y_pred)
print(f'Accuracy: {accuracy}')

sensitivity = recall_score(Y[test], y_pred)
print(f'Sensitivity (Recall): {sensitivity}')

specificity = recall_score(Y[test], y_pred, pos_label=0)
print(f'Specificity: {specificity}')

kappa = cohen_kappa_score(Y[test], y_pred)
print(f'Kappa Score: {kappa}')

f1 = f1_score(Y[test], y_pred)
print(f'F1 Score: {f1}')

roc_auc = roc_auc_score(Y[test], y_pred)
print(f'AUC Score: {roc_auc}')

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrices[-1], annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
