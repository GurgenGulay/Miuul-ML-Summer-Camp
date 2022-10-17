##################################################
# Uygulama - Mülakat Sorusu
##################################################
"""
Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz.

before: "hi my name is john and i am learning python"
after: "Hi NaMe iS JoHn aNd i aM LeArNiNg pYtHoN "
"""

range(len("miuul"))  # range bize iki değer arasında sayı üretme imkanı sağlar
range(0, 5)

for i in range(0, 5):
    print(i)


def alternating(string):
    new_string = ""  # yapılan değişiklikleri buraya kaydetmek istiyoruz o yüzden oluşturduk.
    # girilen string'in index'lerinde gez.
    for string_index in range(len(string)):
        # index çift ise büyük harfe çevir
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        # index tek ise küçük harfe çevir
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("hi my name is john and i am learning python")


############################
# break & continue & while
#############################

salaries = [1000, 2000, 3000, 4000, 5000,]

for salary in salaries:
    if salary == 3000:
        break
    print(salary)


for salary in salaries:
    if salary == 3000:
        continue            # devam et çalışma diğer iterasyona geç
    print(salary)

#while: ...' dı sürece anlamına gelir.

number = 1
while number < 5:   # burası doğru olduğu sürece
    print(number)
    number += 1

#######################################################
# Enumerate: Otomatik Counter/Indexer ile for loop
#######################################################
"""
İteratif yani üzerinden gezilebilir örneğin bir liste içerisinde gezerken bu elemanlara belirli bir işlem uygularken
aynı zamanda işlem uygulanan elemanların index bilgisini de tutup gerekirse bu index bilgisine göre de bir işlem yapmak
istediğimizde hayat kurtaran bir yapıdır.
"""

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)

for index, student in enumerate(students):          # enumerate(hangi_yerde_kullanmak_istiyorsak)
    print(index, student)              #enumerate(students, 1) başlatmak istediğimiz bir sayı varsa bu şekilde yazıyoruz


A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

A
B

#######################################
# Uygulama - Mülakat Sorusu
#######################################
"""
divide_student fonksiyonu yazınız.
Çift indexte yer alan öğrencileri bir listeye alınız.
Tek indexte yer alan öğrencileri başka listeye alınız.
Fakat bu iki liste tek bir liste olarak return olsun.
"""

students = ["John", "Mark", "Venessa", "Mariam", "Cesur", "Aslı", "Cem", "Fatoş", "Şesu", "Gaffur", "Yaprak"]

def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups   # bu fonk. işlenebilir şekilde return etsin

st = divide_students(students)

####################################################
# alternating fonksiyonunun enumerate ile yazılması
####################################################
# Daha önce çözdüğümüz örnekteki gibi tek indextekileri küçük, çift olanları büyük harfle yazdıralım.

def alternating_with_enumerate(string):
    new_string = ""
    for i, letter in enumerate(string):    #hem stringin(elemanlarını değiştirebilmek için) kendisinde hem de indexte gezmem(index çift mi tek mi bakabileyim) gerekiyor
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


alternating_with_enumerate("hi my name is john and i am learning python")


#######################
# Zip
######################
"""
Birbirinden farklı şekilde olan listeleri bir arada değerlendirme imkanı sağlar.
"""

students = ["John", "Mark", "Venessa", "Mariam"]

departments = ["mathematics", "statistics", "physics", "astronomy"]

ages = [23, 30, 26, 22]

list(zip(students, departments))

type(list(zip(students, departments)))

####################################################
# lambda, map, filter, reduce
###################################################
"""
lambda bir fonksiyon tanımlama şeklidir, atama yapılmaz ama biz örnekte uyarı alacağımızı görmek için atama yaptık. 
def'ten farkı bunlar kullan at fonksiyonlardır, apply-map gibi diğer bazı araçlarda kullanıldığında asıl kullanılma 
amacını yerine getirir. Değişkenlerde yer tutmaması için kullanılır.
"""
def summer(a, b):
    return a + b

summer(1, 3) * 9



new_sum = lambda a, b: a + b

new_sum(4, 5)

# map
"""
Döngü yazmaktan kurtarır. İçerisinde gezebileceği iteratif bir nesne ve bu nesneye uygulamak istediğimiz fonk. 
verdiğimizde bu işlemi otomatik olarak yapar.
"""

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

new_salary(1000)


for salary in salaries:
    print(new_salary(salary))


list(map(new_salary, salaries))    # for ile aynısını yaptı tek satırda halletmiş olduk.

list(map(lambda  x: x * 20 / 100 + x, salaries))  #lambda ve mapı birleştirdik görüldüğü gibi ciddi oranda kodu sadeleştirdik.

""" Maaşların karesini almaya çalışalım. lambda x: dediğimiz lambda x'e göre fonk. tanımlıyor."""

list(map(lambda  x: x ** 2, salaries))

# filter
"""
filter filtreleme işlemleri için kullanılır. Elimizde örnekteki gibi bir liste olduğunu düşünelim ve belirli bir koşulu
sağlayanları seçmek istediğimizi, sağlamayanları ise seçmek istemediğimizi düşünelim. lambda ile birlikte kullanılım, 
liste formatında olmasını istediğimiz için başına list ekleyelim
"""
list_score = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda  x: x % 2 == 0, list_score))   # 2'ye bölümünden kalan 0 mı sorgusunu yapıyor. 0 yapanları listede toplayacaktır.

# reduce
"""
yapıyı kullanabilmek için import etmeliyiz
"""

from functools import reduce
list_score = [1, 2, 3, 4]                    # 1+2=3---3+3=6---6+4=10 çıktımız 10 olacaktır
reduce(lambda  a, b: a + b, list_score)


########################################################################################################################
#                                                                                                                      #
#                                           Comprehensions                                                             #
#    **** Çok önemli bir konu ****                                                                                                                  #
########################################################################################################################
"""
Birden fazla kod ve satırla yapılabilecek işlemleri kolayca istediğimiz çıktı veri yapısına göre tek bir satırda 
gerçekleştirme imkanı sağlıyan yapılardır.
"""

###################################
# List Comprehensions
###################################
"""
Örneğin bir liste üzerinde gezip bu elemanlara çeşitli işlemler uygulayıp, bu işlem uygulanmış elemanları tekrar başka 
bir listede görmek istediğimizde burada peş peşe bir çok işlem yapmamız gerekmekte (boş bir liste oluştur var, olan liste
içerisinde gez, elemanlara bir yada daha fazla işlem yap ve sonrasında bu işlemler neticesinde alınan sonuçlar için bir 
liste daha yap ) işte bu tip işlemleri çıktısı liste olacak şekilde elde etme imkanı yakalarız.
"""

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))

null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))


null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))

# buraya kadar yaptığımız uzun yol olandı, şimdi tek satırda halledelim

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]

# şimdide maaşlar listesindeki her bir maaşı 2 ile çarpmak istediğimizi düşünelim
"""
Şimdi ilk olarak listemizi açıyoruz [] sonra for maaş in nerde gezeyim listemizin_adı  bu yakaladığım maaşları napayım 
peki diye soruyor bize python diyoruz ki maaşları 2 ile çarp bu kadar bitti :)
"""

[salary * 2 for salary in salaries]

# şimdi de maaşı 3000'den az olanları 2 ile çarpalım

"""
bu bölüm for salary in salaries maaşlarda gezicek, gezdiğinde her bir maaşı yakaladığında if salary < 3000 bu bölüm 
çalışacak eğer maaş 3000'den küçük ise salary * 2 bunu yapıcak değilse birşey yapmıycak.
"""

[salary * 2 for salary in salaries if salary < 3000]

# maaşı büyük olanları napıcaz tabi ki else yapısını getiririz.

"""
DİKKAT! 
Eğer if bloğunu else olmadan tek başına kullanıcaksak if bloğu sağda for bloğu solda kalır ama
if ve else beraber kullanılacaksa if-else bloğu solda for bloğu sağda yazılır.
"""

[salary * 2  if salary < 3000 else salary * 0 for salary in salaries]

# Diyelim ki elimizde var olan bir fonksiyonu da bu yapıların içerisinde kullanmak istiyoruz. Hadi yapalım

[new_salary(salary * 2)  if salary < 3000 else salary * 0 for salary in salaries]

"""
Gördüğümüz gibi 3000'den küçük olanları hem ikiyle çarptı hem de belirli bir oranda zam yaptı hadı else bloğuna da 
fonksiyonumuzu ekleyelim. (0'la çarpılmasının anlamı olmadığı için 0.2 ile güncelledik)  
"""

[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2 ) for salary in salaries]


# Yeni birşeyler daha deneyelim.
"""
Diyelim ki elimizde 2 tane liste var. Biri istediğimiz öğrenciler diğeri ise istemediğimiz öğrencilerin listesi olsun.
İstediğim öğrencilerin listesindeki isimleri büyük, istemediğim öğrencilerin isimlerini ise küçük harfle değiştirelim.
"""

students = ["John", "Mark", "Venessa", "Mariam"]

students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]


# not in yapısı

[student.upper() if student not in students_no else student.lower() for student in students]


###########################################
# Dict Comprehensions
##########################################

dictionary = {'a': 1,
              'b': 2,
              'c': 3,
              'd': 4,}

dictionary.keys()
dictionary.values()
dictionary.items() # items metodu ile erişebiliyoruz:  çıktığımızın liste olduğunu köşeli parantezlerden anlıyoruz aynı zamanda her bir elemanı da tuple.


# Diyelim ki key'lere dokunmak istemiyoruz ve her value değerinin karesini almak istiyoruz.

{k: v ** 2 for (k, v) in dictionary.items()}

# eğer key'ler içinde işlem yapmak istersek

{k.upper(): v for (k, v) in dictionary.items()}

# ikisini aynı anda yapalım

{k.upper(): v ** 2 for (k, v) in dictionary.items()}


########################################
# Uygulama - Mülakat Sorusu
#######################################
""" 
Amaç : çift sayıların karesi alınarak bir sözlüğe eklenmek istenmektedir.
Key'ler orjinal değerler value'lar ise değiştirilmiş değerler olacak.
"""
# ilk eski yöntemle yapalım.

numbers = range(10)   # 0'dan 10' kadar sayıları ifade etmektedir.
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2   # bu kısım önemli bir kullanım.Eşittirden öncesi key, eşittirden sonrası ise value olucaktır.

new_dict

# Şimdi comprehension yapısı ile yazalım
# Not! Tek bir if yapısı olduğu için sağ tarafta yer alıcaktır.

{n: n ** 2 for n in numbers if n % 2 == 0}


#####################################################################################
# List & Dict Comprehension Uygulamalar
######################################################################################

#####################################################
# Bir Veri Setindeki Değişken İsimlerini Değiştirmek
#####################################################
"""
before:
['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium', 'ins_losses', 'abbrev']
after:
['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSSES', 'ABBREV']
"""
# İlk klasik yöntemle yapalım.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())

A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A     # veri setimizi de değiştirmiş olduk ve isteneni yaptık


# Şimdi comprehension yapısı ile yazalım

df = sns.load_dataset("car_crashes")

df.columns = [col.upper() for col in df.columns]

df.columns


########################################################################################
# İsminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG eklemek istiyoruz.
############################################################################################
"""
before: 
['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSSES', 'ABBREV']
after:
['NO_FLAG_TOTAL', 'NO_FLAG_SPEEDING', 'NO_FLAG_ALCOHOL', 'NO_FLAG_NOT_DISTRACTED', 'NO_FLAG_NO_PREVIOUS', 'FLAG_INS_PREMIUM', 'FLAG_INS_LOSSES', 'NO_FLAG_ABBREV']
"""

[col for col in df.columns if "INS" in col]  # INS olanların listesini getirdi. Dedikti df.columns da gez ve ıns olan kolonları getir

["FLAG_" + col for col in df.columns if "INS" in col]  # INS olan kolonların flag ekle dedik. Hatırlatma A + B yazdığımızda çıktısı AB olur.

# Şimdi sağlamayan durum olan NO_FLAG yazısını else bloğu ile yazdıralım

["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns ]

# peki bu değiştirdiğimiz dataframe kalıcı hale getirebilir miyiz? Haydi yapalım.

df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns ]


#####################################################################################
# Amaç : key'i string, value'su aşağıdaki gibi bir liste olan sözlük oluşturmak.
# Sadece sayısal değişkenler için yapmak istiyoruz.
#####################################################################################
"""
Output:
{'total': ['mean', 'min', 'max', 'sum'],
 'speeding': ['mean', 'min', 'max', 'sum'],
 'alcohol': ['mean', 'min', 'max', 'sum'],
 'not_distracted': ['mean', 'min', 'max', 'sum'],
 'no_previous': ['mean', 'min', 'max', 'sum'],
 'ins_premium': ['mean', 'min', 'max', 'sum'],
 'ins_losses': ['mean', 'min', 'max', 'sum']}
"""

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

# df[col] dataframde ilgili değişken seçilecek .dtype != "O"  object(yani kategorik değişkenleri temsil eder) olmayan tipteki değişkenleri seçmiş olucak
# dtype == "O" deseydik object olanları yani string tipteki değişkenleri getirir ama değildir kullandığımızda ise bize nümerik tipteki değişkenleri getirir
num_cols = [col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list = ["mean", "min", "max", "sum"]

for col in num_cols:
    soz[col] = agg_list

# kısa yol
new_dict = {col: agg_list for col in num_cols}

"""
Eğer bir dataframe köşeli parantez girdikten sonra bir değişken listesi verirseniz bu df içerisinden o değişkenleri seçer. örnekte yaptığımızda sadece sayısal değişkenler kalır
"""

df[num_cols].head()

df[num_cols].agg(new_dict)


