import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Eksek veri tamamlama
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


# Train-test ayırma
x_train, x_test, y_train, y_test = train_test_split(
    sonveriler.iloc[:, :-1], sonveriler.iloc[:, -1], test_size=0.33, random_state=0
)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=5)
grid.fit(X_train, y_train)

# En iyi parametre
best_params = grid.best_params_
print("Best parameters:", best_params)

# 5 kat çapraz doğ.
best_estimator = grid.best_estimator_
cv_scores = cross_val_score(best_estimator, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", np.mean(cv_scores))

# Tahmin
y_pred = best_estimator.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_estimator.decision_function(X_test))
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DVM ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Performans Metrikleri
print("Classification Report:\n", classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

sensitivity = recall_score(y_test, y_pred)
print(f'Sensitivity (Recall): {sensitivity}')

specificity = recall_score(y_test, y_pred, pos_label=0)
print(f'Specificity: {specificity}')

kappa = cohen_kappa_score(y_test, y_pred)
print(f'Kappa Katsayısı: {kappa}')

f1 = f1_score(y_test, y_pred)
print(f'F-ölçümü: {f1}')

roc_auc = roc_auc_score(y_test, y_pred)
print(f'AUC Katsayısı: {roc_auc}')
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, recall_score, cohen_kappa_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import seaborn as sns

# Load data
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

# Independent and dependent variables
x = sonveriler.iloc[:, :-1].values
y = sonveriler.iloc[:, -1].values

# Addressing class imbalance (using SMOTE)
smote = SMOTE(random_state=0)
x_resampled, y_resampled = smote.fit_resample(x, y)

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x_resampled, y_resampled, test_size=0.33, random_state=0
)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# SVM with GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=cv)
grid.fit(X_train, y_train)

# Best parameters
best_estimator = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Predictions
y_pred = best_estimator.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_estimator.decision_function(X_test))
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Classification report and metrics
print("Classification Report:\n", classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

sensitivity = recall_score(y_test, y_pred)
print(f'Sensitivity (Recall): {sensitivity}')

specificity = recall_score(y_test, y_pred, pos_label=0)
print(f'Specificity: {specificity}')

kappa = cohen_kappa_score(y_test, y_pred)
print(f'Kappa Score: {kappa}')

f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1}')

roc_auc = roc_auc_score(y_test, y_pred)
print(f'AUC Score: {roc_auc}')

'''







