# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:51:27 2023

@author: iyagm
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('diabetes.csv')
print(veriler)

insülin=veriler.iloc[:,4:5].values
cevap = veriler.iloc[:,-1].values
print(cevap)

insülin0 = veriler[insülin ==0]
cevap0 = veriler[cevap == 0]
cevap1 = veriler[cevap == 1]

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

#cevaps_0 = veriler_son[cevap_s == 0]
cevaps_0 = veriler_son[cevap_s == 0]

# Cevap_0 DataFrame'indeki veri sayısını yazdır
print("Cevap 0 olan veri sayısı:", len(cevaps_0))

# cevap1 olan ve insülin 0 olan satırları bul
sil_sart = (veriler_son['Outcome'] == 1) & (veriler_son['Insulin'] == 0)

# belirtilen koşulu sağlayan ilk 4 satırı sil
veriler_sonn = veriler_son.drop(veriler_son[sil_sart].index[:4])
print("verilerolan veri sayısı:", len(veriler_sonn))

# reset_index kullanarak indexleri sıfırla
veriler_sifirlanmis = veriler_sonn.reset_index(drop=True)

# sıfırlanmış verileri yazdır
print(veriler_sifirlanmis)

'''
df = pd.DataFrame(veriler_sifirlanmis)

# DataFrame'i CSV dosyasına kaydet
df.to_csv('veriler_isleme.csv', index=False)
'''
