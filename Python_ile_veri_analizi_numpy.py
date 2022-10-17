########################################################################################################################
#                                   NUMPY                                                                              #
########################################################################################################################
"""
Numpy'ın sadece pythonda ki numerik işlemlerin temelini oluşturuyor olduğu bilgisi
genelde veri analitiği, veri analizi gibi işlemlerde pandas kullanılır fakat pandasta numpy'ın üzerine kurulduğu için
numpy'ı bilmek gerekmektedir.
"""
#     Neden NumPy ?      #
# 1. Hızlı, hızın sebebi verimli veri saklamadır. Nasıl yani dersekte sbt tipte veri saklar ve bundan dolayıda hızlı bir
# şekilde çalışır.

# 2. Fonksiyonel düzeyde / vektörel düzeyde / yüksek seviyede çeşitli kolaylıklar sağlar.
"""
Verimli veri saklama
yüksek seviyeden işlemlerdir (vektörel işlemlerdir)
listelere kıyasla çok daha hızlı işlem yapar.(sabit tipte veri tutarak yapıyor)
döngü yazmaya gerek olmadan array seviyesinde çok basit işlemlerle normalde daha çok çaba gerektiren işlemleri
gerçekleştirme imkanı sağlar
"""
#########################
# Neden Numpy?
########################

import numpy as np
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i] * b[i])

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

#################################################################
# Numpy Array'i Oluşturmak (Creating Numpy Arrays)
#################################################################

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)

np.random.randint(0, 10, size=10)   # alt sınır, üst sınır, boyut

np.random.normal(10, 4, (3, 4))  # ortalama,argüman,boyut bilgisi / ort=10, standart sapması= 4, (3, 4)'lük matris

############################################################
# Numpy Array Özellikleri (Attibutes of Numpy Arrays)
###########################################################

"""
ndim: boyut sayısı
shape: boyut bilgisi
size: toplam eleman sayısı
dtype: array veri tipi
"""

a = np.random.randint(10, size=5)  # başlangıç girmediğimiz için 0'dan 10'a kadar olucak, size 5 old için 5 tane
a.ndim  # tek boyutlu olduğu için 1 geldi
a.shape # (5, ) tek boyutlu ve içerisinde 5 tane eleman var dedi
a.size  # toplam eleman sayısını vermektedir
a.dtype  # int32 olarak cevap verdi

#####################################################
# Yeniden Şekillendirme ( Reshaping )
#####################################################

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3)

########################################
# Index Seçimi ( Index Selections )
########################################

a = np.random.randint(10, size=10)
a[0]
a[0:5]   # 0'dan 5. indexe kadar yazdır
a[0] = 99
a

m = np.random.randint(10, size=(3, 5))
m[0, 1]   #satır,sutün

m[2, 3] = 999
m

# eğer int değil de float bir ifade eklemek isteseydim
m[2, 3] = 2.9
m  # 2 olarak değiştirmiş çünkü numpy fix type yani sabit tipte arraydir. Diğer değerlerimi int idi.

m[:, 0]  # bütün satırları seç 0. sutünu seç
m[1, :]  # 1. satırın tüm sutünları
m[0:2, 0:3]  # satırdan 0'dan 2'ye kadar git, sutünlarda 0'dan 3'e kadar git

######################################################################
# Fancy Index
######################################################################

v = np.arange(0, 30, 3)  # 0'dan 30'a kadar(30 hariç) 3'er 3'er artıcak şekilde
v
v[1]
v[4]

catch = [1, 2, 3]

v[catch]

######################################################
# Numpy'da Koşullu İşlemler ( Conditions on NUmpy )
#######################################################

v = np.array([1, 2, 3, 4, 5])

##################################
# Klasik döngü ile
#################################

ab = []

for i in v:
    if i < 3:
        ab.append(i)

#################################
# Numpy ile
################################
v < 3

v[v < 3]
v[v > 3]
v[v != 3]
v[v == 3]
v[v >= 3]

#################################################################
# Matematiksel İşlemler ( Mathematical Operations )
################################################################

v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1

"""
Yukarıda operatörler ile yaptık biz bu işlemleri metodlar ile de yapabiliriz
"""

np.subtract(v, 1)   # çıkarma işlemi
np.add(v, 1)  # toplama
np.mean(v)  # ortalama
np.sum(v)   # toplam alma
np.min(v)    # min
np.max(v)   # max
np.var(v)   # varyans

# türev, integral gibi ya da iki bilinmeyenli denklemleri çözmek gibi işlemlerde gerçekleştirebiliriz.

#####################################################
# Numpy ile iki bilinmeyenli denklem çözümü
####################################################

"""
5*x0 + x1 = 12
  x0 + 3*x1 = 10
"""

a = np.array([[5, 1], [1, 3]])   # [[xo'ın katsayıları], [x1'in katsayıları]]
b = np.array([12, 10])       # denklemlerin çözümleri

np.linalg.solve(a,b)
