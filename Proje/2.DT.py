import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, cohen_kappa_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

veriler = pd.read_csv('veriler_isleme.csv')
print(veriler)

eksik_veri = veriler.iloc[:, 1:6]

# Eksik Veri
df = pd.DataFrame(eksik_veri)
# 0 olan değerleri NaN ile değiştir
df.replace(0, np.nan, inplace=True)
# Her sütunun ortalamasını bul
ortalama = df.mean()
# Sütunlardaki NaN değerleri sırasıyla ortalama ile doldur
for column in df.columns:
    df[column].fillna(ortalama[column], inplace=True)


# Birleştir
# doğumları ayrı alıp dataframe yapıldı
doğ = veriler.iloc[:, :1]
doğ = pd.DataFrame(doğ)
sonveriler = pd.concat([doğ, df], axis=1)
sonveriler = pd.concat([sonveriler, veriler.iloc[:, -3:]], axis=1)

# train ve test ayrımı
x_train, x_test, y_train, y_test = train_test_split(
    sonveriler.iloc[:, :-1], sonveriler.iloc[:, -1], test_size=0.33, random_state=0
)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

print("Cevap train olan veri sayısı:", len(x_train))
print("Cevap test olan veri sayısı:", len(x_test))

# Karar ağacı için kullanılacak parametreler ve değer aralıkları
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# DecisionTreeClassifier
dtc = DecisionTreeClassifier()

# GridSearchCV ile en iyi parametreleri bulma
grid_search = GridSearchCV(dtc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# En iyi parametreleri yazdırma
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# En iyi tahmin modelini seçme
best_estimator = grid_search.best_estimator_

# Cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(best_estimator, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", np.mean(cv_scores))

# Test seti üzerinde tahmin yapma
y_pred_grid = best_estimator.predict(X_test)

# Karmaşıklık Matrisi
cm_grid = confusion_matrix(y_test, y_pred_grid)
print("Confusion Matrix:")
print(cm_grid)
plt.figure(figsize=(5, 5))
sns.heatmap(cm_grid, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix with GridSearchCV')
plt.show()

#ROC eğrisi
fpr, tpr, thresholds = roc_curve(y_test, y_pred_grid)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.title('KA ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#Performans Metrikleri
# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred_grid))

# Accuracy
accuracy_grid = accuracy_score(y_test, y_pred_grid)
print(f'Accuracy with GridSearchCV: {accuracy_grid}')

# Sensitivity (Recall)
sensitivity_grid = recall_score(y_test, y_pred_grid)
print(f'Sensitivity (Recall) with GridSearchCV: {sensitivity_grid}')

# Specificity
specificity_grid = recall_score(y_test, y_pred_grid, pos_label=0)
print(f'Specificity with GridSearchCV: {specificity_grid}')

# Kappa Katsayısı
kappa_grid = cohen_kappa_score(y_test, y_pred_grid)
print(f'Kappa Katsayısı with GridSearchCV: {kappa_grid}')

# F-ölçümü
f1_grid = f1_score(y_test, y_pred_grid)
print(f'F-ölçümü with GridSearchCV: {f1_grid}')

# AUC Katsayısı
roc_auc_grid = roc_auc_score(y_test, y_pred_grid)
print(f'AUC Katsayısı with GridSearchCV: {roc_auc_grid}')



