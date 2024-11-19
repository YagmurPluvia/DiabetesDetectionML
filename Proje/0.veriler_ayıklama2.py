import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('diabetes.csv')
print(veriler)

insülin=veriler.iloc[:,4:5].values
cevap = veriler.iloc[:,-1].values
print(cevap)

insülin0 = veriler[insülin ==0]    #Boş Veri sayısı
cevap0 = veriler[cevap == 0]       #Sağlıklı
cevap1 = veriler[cevap == 1]       #Hasta

# Cevap_1_verileri DataFrame'indeki veri sayısını yazdır
print("Cevap 0 olan veri sayısı:", len(cevap0))
print("Cevap 1 olan veri sayısı:", len(cevap1))
print("İnsülin 0 olan veri sayısı:", len(insülin0))

# İnsülin sütunu değeri 0 olan ve çıktısı 0 olan satırları filtrele
veriler_filtrerli = veriler[(veriler['Insulin'] == 0) & (veriler['Outcome'] == 0)]
print("verilerolan veri sayısı:", len(veriler_filtrerli))

# Filtrelenmiş satırları orijinal verilerden çıkar
veriler_son = veriler.drop(veriler_filtrerli.index)
cevap_s = veriler_son.iloc[:,-1].values

#Sağlıklı sayısı çıkarıldıktan sonra bulma
#cevaps_0 = veriler_son[cevap_s == 0]
cevaps_0 = veriler_son[cevap_s == 0]
cevaps_1 = veriler_son[cevap_s == 1]

# Cevap_0 DataFrame'indeki veri sayısını yazdır
print("Cevap 0 olan veri sayısı:", len(cevaps_0))
print("Cevap 1 olan veri sayısı:", len(cevaps_1))

# cevap 1(hasta) olan ve insülin 0 olan satırları bul
sil_sart = (veriler_son['Outcome'] == 1) & (veriler_son['Insulin'] == 0)

# belirtilen koşulu sağlayan ilk 4 satırı sil  --->Eşitleme yapmak için
veriler_sonn = veriler_son.drop(veriler_son[sil_sart].index[:4])
print("verilerolan veri sayısı:", len(veriler_sonn))

# reset_index kullanarak indexleri sıfırla
veriler_sifirlanmis = veriler_sonn.reset_index(drop=True)

# sıfırlanmış verileri yazdır
print(veriler_sifirlanmis)

#Eksik veri doldurma
eksik_veri = veriler_sifirlanmis.iloc[:, 1:6] #Glucose dan BMI ya kadar al
df = pd.DataFrame(eksik_veri)
df.replace(0, np.nan, inplace=True)
ortalama = df.mean()
for column in df.columns:
    df[column].fillna(ortalama[column], inplace=True)

#Veileri geri ekleyip dataframe yi oluşturma
doğ = veriler_sifirlanmis.iloc[:, :1]
sonveriler = pd.concat([doğ, df], axis=1)
sonveriler = pd.concat([sonveriler, veriler.iloc[:, -3:]], axis=1)

#Grafikler

plt.plot(veriler['Glucose'], label='Veriler', color='blue')
plt.plot(sonveriler['Glucose'], label='Son Veriler', color='red')
plt.title('Glucose Özelliği Karşılaştırması')
plt.xlabel('Index')
plt.ylabel('Glucose Değeri')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(veriler['Insulin'], label='Veriler', color='blue')
plt.plot(sonveriler['Insulin'], label='Son Veriler', color='red')
plt.title('Insulin Özelliği Karşılaştırması')
plt.xlabel('Index')
plt.ylabel('Insulin Değeri')
plt.legend()
plt.grid(True)
plt.show()


# Veriler ve son veriler arasındaki değişimi görselleştirmek için bir kutu grafiği oluştur
plt.figure(figsize=(10, 6))
veriler.boxplot(color='blue', positions=np.array(range(len(veriler.columns))) * 2.0, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightblue'))
sonveriler.boxplot(color='red', positions=np.array(range(len(veriler.columns))) * 2.0 + 0.4, widths=0.4, patch_artist=True, boxprops=dict(facecolor='lightcoral'))

# Grafik özelliklerini düzenleme
plt.xticks(range(0, len(veriler.columns) * 2, 2), veriler.columns, rotation=90)
plt.title('Veriler ve Son Veriler Arasındaki Değişim')
plt.xlabel('Özellikler')
plt.ylabel('Değerler')
plt.legend(['Veriler', 'Son Veriler'])
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


# Veriler ve son veriler arasındaki değişimi görselleştirmek için çubuk grafikleri oluştur
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(veriler.columns))
opacity = 0.8

plt.bar(index, veriler.mean(), bar_width, alpha=opacity, color='b', label='Veriler')
plt.bar(index + bar_width, sonveriler.mean(), bar_width, alpha=opacity, color='r', label='Son Veriler')

# Grafik özelliklerini düzenleme
plt.xlabel('Özellikler')
plt.ylabel('Ortalama Değerler')
plt.title('Veriler ve Son Veriler Arasındaki Değişim')
plt.xticks(index + bar_width, veriler.columns, rotation=90)
plt.legend()
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()



plt.hist(sonveriler['Glucose'], bins=20, color='skyblue', edgecolor='black')
plt.title('Glucose Dağılımı')
plt.xlabel('Glucose Değeri')
plt.ylabel('Frekans')
plt.grid(True)
plt.show()

plt.scatter(sonveriler['Glucose'], sonveriler['BMI'], color='green', alpha=0.5)
plt.title('Glucose ve BMI Arasındaki İlişki')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

# Sütunları ayırma
float_columns = [column for column in veriler.columns if veriler[column].dtype == 'float64']
binary_columns = [column for column in veriler.columns if veriler[column].dtype == 'int64']

# Grafikleri oluşturma
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Float sütunlar için çubuk grafikleri
axs[0].set_title('Float Özellikler için Veriler ve Son Veriler Arasındaki Değişim')
for i, column in enumerate(float_columns):
    axs[0].bar(i, veriler[column].mean(), bar_width, alpha=opacity, color='b', label='Veriler')
    axs[0].bar(i + bar_width, sonveriler[column].mean(), bar_width, alpha=opacity, color='r', label='Son Veriler')

axs[0].set_xlabel('Float Özellikler')
axs[0].set_ylabel('Ortalama Değerler')
axs[0].set_xticks(range(len(float_columns)))
axs[0].set_xticklabels(float_columns, rotation=90)
axs[0].legend()
axs[0].grid(True, axis='y')

# Binary sütunlar için çubuk grafikleri
axs[1].set_title('Binary Özellikler için Veriler ve Son Veriler Arasındaki Değişim')
for i, column in enumerate(binary_columns):
    axs[1].bar(i, veriler[column].mean(), bar_width, alpha=opacity, color='b', label='Veriler')
    axs[1].bar(i + bar_width, sonveriler[column].mean(), bar_width, alpha=opacity, color='r', label='Son Veriler')

axs[1].set_xlabel('Binary Özellikler')
axs[1].set_ylabel('Ortalama Değerler')
axs[1].set_xticks(range(len(binary_columns)))
axs[1].set_xticklabels(binary_columns, rotation=90)
axs[1].legend()
axs[1].grid(True, axis='y')

plt.tight_layout()
plt.show()




'''
df = pd.DataFrame(veriler_sifirlanmis)

# DataFrame'i CSV dosyasına kaydet
df.to_csv('veriler_isleme.csv', index=False)
'''
