# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

#from sklearn.impute import SimpleImputer

veriler = pd.read_csv('veriler_isleme.csv')
print(veriler)

eksik_veri =veriler.iloc[:,1:6]

#Eksik Veri
df = pd.DataFrame(eksik_veri)
# 0 olan değerleri NaN ile değiştir
df.replace(0, np.nan, inplace=True)
# Her sütunun ortalamasını bul
ortalama = df.mean()
# Sütunlardaki NaN değerleri sırasıyla ortalama ile doldur
for column in df.columns:
    df[column].fillna(ortalama[column], inplace=True)

# Sonuçları yazdır
#print("Original DataFrame:")
#print(df)

#Birleştir
#doğumları ayrı alıp dataframe yapıldı
doğ = veriler.iloc[:,:1]
doğ = pd.DataFrame(doğ)
sonveriler = pd.concat([doğ,df],axis=1)
sonveriler = pd.concat([sonveriler, veriler.iloc[:,-3:]],axis=1)

#train ve test ayrımı
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1],test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

print("Cevap train olan veri sayısı:", len(x_train))
print("Cevap test olan veri sayısı:", len(x_test))


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='entropy')

# 5 kat çapraz doğrulama
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rfc, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", np.mean(cv_scores))

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

#Karmaşıklık Matrixsi---------------------------------------------------
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.show()


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, rfc.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('RO ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Performans Metrikleri
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Sensitivity (Recall)
sensitivity = recall_score(y_test, y_pred)
print(f'Sensitivity (Recall): {sensitivity}')

# Specificity
specificity = recall_score(y_test, y_pred, pos_label=0)
print(f'Specificity: {specificity}')

# Kappa Katsayısı
kappa = cohen_kappa_score(y_test, y_pred)
print(f'Kappa Katsayısı: {kappa}')

# F-ölçümü
f1 = f1_score(y_test, y_pred)
print(f'F-ölçümü: {f1}')

# AUC Katsayısı
roc_auc = roc_auc_score(y_test, y_pred)
print(f'AUC Katsayısı: {roc_auc}')
