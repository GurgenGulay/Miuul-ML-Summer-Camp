###############
# Görev 1:
###############
# Verilen değerlerin veri yapılarını inceleyiniz

x = 8
y = 3.2
z = 8j + 18
a = "Hello World"
b = True
c = 23 < 22
l = [1, 2, 3, 4]
d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
t = ("Machine Learning", "Data Science")
s = {"Python", "Machine Learning", "Data Science"}
type(s)

###############
# Görev 2:
###############
"""
Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
text = "The goal is to turn data into information, and information into insight."
Beklenen çıktı: ['THE', 'GOAL', 'IS', 'TO', 'TURN', 'DATA', 'INTO', 'INFORMATION,', 'AND', 'INFORMATION', 'INTO', 'INSIGHT.']
"""

text = "The goal is to turn data into information, and information into insight."

upper = text.upper()
print(upper.split(" "))

# Şimdi comprehension yapısı ile yazalım
upper = text.upper()
[upper.split(" ") for index in text]

###############
# Görev 3:
###############
# Verilen listeye aşağıdaki adımları uygulayınız
"""
lst = ["D","A","T","A","S","C","I","E","N","C","E"]
Adım 1: Verilen listenin eleman sayısına bakınız.
Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
Adım 4: Sekizinci indeksteki elemanı siliniz.
Adım 5: Yeni bir eleman ekleyiniz.
Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
"""
lst = ["D","A","T","A","S","C","I","E","N","C","E"]
#Adım 1: Verilen listenin eleman sayısına bakınız.
print(len(lst))

#Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
print(lst[0], lst[10])

#Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
newList = lst[0:4]
print(newList)

#Adım 4: Sekizinci indeksteki elemanı siliniz.
lst.pop(8)
print(lst)

#Adım 5: Yeni bir eleman ekleyiniz.
lst.append(1)
print(lst)

#Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")
print(lst)

###############
# Görev 4:
###############
# Verilen sözlük yapısına aşağıdaki adımları uygulayınız

dict = {'Christian': ["America", 18],
        "Daisy": ["England", 12],
        "Antonio": ["Spain", 22],
        "Dante": ["Italy", 25]}

# Adım 1: Key değerlerine erişiniz.
print(dict.keys())

# Adım 2: Value'lara erişiniz.
print(dict.values())

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict["Daisy"][1] = 13
print(dict)

# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict["Ahmet"] = ["Turkey", 24]
print(dict)

# Adım 5: Antonio'yu dictionary'den siliniz.
dict.pop("Antonio")
print(dict)

###############
# Görev 5:
###############
"""
Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri return 
eden fonksiyon yazınız.
"""

l = [2, 13, 18, 93, 22]

def oddEven(myList):
    #output = [[], []]
    odd = []
    even = []
    for i in myList:
        if i % 2 == 0:
            even.append(i)
        else:
            odd.append(i)

    return even, odd

even, odd = oddEven(l)

print(even)
print(odd)

###############
# Görev 6:
###############
"""
List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve 
başına NUM ekleyiniz
Not: Numeric olmayan değişkenlerin de isimleri büyümeli. Tek bir list comprehension yapısı kullanılmalı.
"""
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

a = ["NUM_" + col.upper() for col in df.columns]

###############
# Görev 7:
###############
"""
List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan değişkenlerin isimlerinin sonuna 
"FLAG" yazınız.
Not: Numeric olmayan değişkenlerin de isimleri büyümeli. Tek bir list comprehension yapısı kullanılmalı.
"""

b = [col.upper() if "no" in col else col.upper() + "_FLAG" for col in df.columns]

###############
# Görev 8:
###############
"""
List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz 
ve yeni bir dataframe oluşturunuz

og_list = ["abbrev", "no_previous"]

Beklenen çıktı:
    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
0    18.8     7.332    5.640          18.048       784.55      145.08
1    18.1     7.421    4.525          16.290      1053.48      133.93
2    18.6     6.510    5.208          15.624       899.47      110.35
3    22.4     4.032    5.824          21.056       827.34      142.39
4    12.0     4.200    3.360          10.920       878.41      165.63

Notlar: 
Önce verilen listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
Sonra df[new_cols] ile bu değişkenleri seçerek yeni bir df oluşturunuz ve adını new_df olarak isimlendiriniz.
"""
og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]

