# 1. Kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import seaborn as sns  # Eksik kütüphane

# Veriyi okuma
veriler = pd.read_csv('son_veri.csv')
print(veriler)

# Bağımsız ve bağımlı değişkenler
x = veriler.iloc[:, :-1].values
y = veriler.iloc[:, -1].values

# Model oluşturma ve eğitme
ensemble_model = VotingClassifier(estimators=[
    ('random_forest', RandomForestClassifier(n_estimators=100, random_state=0)),
    ('gradient_boosting', GradientBoostingClassifier(n_estimators=100, random_state=0)),
    ('xgboost', XGBClassifier(n_estimators=100, random_state=0, eval_metric='logloss'))
], voting='soft')

# Sınıflar arasındaki dengesizliği ele alma (SMOTE kullanımı)
smote = SMOTE(random_state=0)
x_resampled, y_resampled = smote.fit_resample(x, y)

# 5 kat çapraz doğrulama
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_scores_ensemble = cross_val_score(ensemble_model, x_resampled, y_resampled, cv=cv, scoring='accuracy')

# Modelin performansını yazdırma
print("Ensemble Model Cross-Validation Scores:", cv_scores_ensemble)
print("Mean Ensemble Model Cross-Validation Score:", np.mean(cv_scores_ensemble))

# Test seti üzerinde tahmin yapma
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.33, random_state=0)
ensemble_model.fit(x_train, y_train)
y_pred = ensemble_model.predict(x_test)

# Karmaşıklık matrisi
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # sns eklendi
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_scores = ensemble_model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Oybirliği Enaemble ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Sınıflandırma raporu ve metrikleri
print("Classification Report:\n", classification_report(y_test, y_pred))

# Performans metrikleri
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