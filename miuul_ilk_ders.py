########################################################################################################################
#                                                                                                                      #
#                                        Çalışma Ortamı Ayarları                                                       #
#                                                                                                                      #
########################################################################################################################

#####################################################################################
# Virtual Environment: İzole çalışma ortamları oluşturmak için kullanılan araçlardır.
# Farklı çalışmalar için oluşabilecek farklı kütüphane ve versiyon ihtiyaçlarını
# çalışmalar birbirini etkilemeyecek şekilde oluşturma imkanı sağlar.
######################################################################################

################################################################################################
# Package Management
# Paket yönetim araçları ?  1.pip  2.pipenv  3.conda(hem sanal ortam hem de paket yöneticisidir)
# Not: venv ve virtualenv paket yönetim aracı olarak pip'i kullanıyor.
#      conda ve pipenv hem paket hem de virtual environment yönetimi yapabiliyor.
#      pip sadece paket yönetimi için kullanılabilir.
################################################################################################


###################################################################################################################
# Sanal ortamların listelenmesi:
# PyCharm içindeki terminalde conda env list  komutunu kullanıyoruz.
# Kendimiz nasıl sanal ortam yaparız?
# Örneğin myenv adında yapalım bir tane komutumuz --> conda create -n myenv
# Sanal ortamı aktif etmek için --> conda activate myenv
# Deactive yapmak için --> conda deactivate
# Yüklü paketlerin listelenmesi için --> conda list
# Paket yüklemek için --> conda install numpy
# Birden fazla paket yüklemek için --> conda install numpy scipy pandas
# Paket silme --> conda remove pandas
# Belirli bir versiyona göre paket yükleme --> conda install numpy=1.20.1    Not: pip de 2 tane eşittir kullanılır.
# Paket yükseltme --> conda upgrade numpy
# Tüm paketleri yükseltmek için --> conda upgrade -all
####################################################################################################################

# pip: pypi (python package index) paket yönetim aracı
# piple yükleme yapmak için --> pip install paket_adı
# Paket yükleme versiyona göre --> pip install pandas==1.2.1
# Kullandığımız kütüphaneleri başka bir çalışma ortamına aktarma --> conda env export > enviroment.yaml (yada yml)
# pip ile yaparken uzantı .txt oluyor.
# içe aktamak için de --> conda env create -f environment.yaml
####################################################################################################################


########################################################################################################################
#                                                                                                                      #
#                                       Data Structures (Veri Yapıları)                                                #
#                                                                                                                      #
########################################################################################################################
# - Veri Yapılanıra Giriş ve Hızlı Özet

x = 46  # integer
type(x)  # tip sorgulama için type() fonksiyonu kullanılır.

x = 10.3  # float
type(x)

x = 2j + 1  # complex sayılar
type(x)

x = "Hello ai era"  # string
type(x)

# Boolean / TRUE FALSE
True
type(True)
5 == 4

# Liste
x = ["btc", "eth", "xrp"]
type(x)

# sözlük  Not: : dan önceki key sonrakiler value'dur.
x = {"name": "Peter", "Age": 36}
type(x)

# Tuple / demet
x = ("python", "ml", "ds")
type(x)

# Set  / key value durumu yoktur
x = {"python", "ml", "ds"}
type(x)

# Not: Liste, tuple, set ve dictionary veri yapıları aynı zamanda Python Collections (Arrays) olarak geçmektedir.

########################################
# Tipleri değiştirmek
#######################################
a = 4
b = 19.3

float(a)
int(b)
int(a * b / 10)

c = a * b / 10
int(c)
#######################################
# - Karakter Dizileri (Strings): str
#######################################
# Bir program yada bir fonksiyon yazıyor isek ve bu programda ekrana bir bilgi paylaşmak istiyorsak print kullanmak zorundayız.

# Çok satırlı karakter dizileri / birden fazla tırnak kullanarak yapabiliriz ve atama işlemi de yapalım örnekte

long_str = """Veri yapıları: Hızlı özet,
Sayılar (Numbers): int, float, complex,
Karakter dizileri (strings): str,
List, Dict, Tuple, Set,
Boolean (TRUE-FALSE): bool"""
print(long_str)

# indexler 0'dan başlar

################################
# Karakter Dizilerinde Slice İşlemi
#####################################
# name[0:2]    #0'dan 2'ye kadar git.2 dahil değil

#############################################
# Stringler içinde arama yapma
############################################
long_str
"veri" in long_str  # büyük-küçük harf hassasiyeti vardır.

"Veri" in long_str

#####################################################
# Sting (Karakte Dizisi) Metodları
#####################################################
# dir ile kullanılabilecek metodları listeleyebiliriz
dir(str)

# len    / stringlerde boyut bilgisine ulaşmak için kullanılır.

name = "John"
type(name)
type(len)

len(name)
len("Gulay Gürgen")

# Method mu yoksa fonksiyon mu nasıl ayırt ederiz?
# Bir fonk class yapısı içinde tanımlanırsa method denir, eğer bir class yapısı içinde değil ise fonksiyondur.

###################################################
# upper() & lower() : küçük-büyük dönüşümleri
##################################################

"miuul".upper()
"MIUUL".lower()

# type(upper)   pyCharm nesne mi method mu ayrımı yapabiliyor bu yüzden biz kodu çalıştırdığımızda hata alıyoruz.
# type(upper())

#######################################
# replace: karakter değiştirir
######################################

hi = "Hello AI Era"
hi.replace("l", "p")  # l harfini p ile değiştir dedik

#################################
# split: böler
################################

"Hello AI Era".split()  # ön tanımlı değeri boşluktur, boşluklara göre böler.

################################
# strip
###############################

" ofofo ".strip()
"ofofo".strip("o")

##############################
# capitalize: ilk harfi büyütür
###############################

"foo".capitalize()

##################################
# Liste (List)
################################
# Değiştirilebilir
# Sıralıdır. Index işlemleri yapılabilir.
# Kapsayıcıdır. Yani içerisinde birden fazla veri yapısını aynı anda tutabilir

notes = [1, 2, 3, 4]
type(notes)
names = ["a", "b", "v", "d"]
not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]]

not_nam[0]
not_nam[6][1]

not_nam[0:4]  # liste elemanlarını 0'dan 4. indexe kadar göster
# değiştirilebilir olduğuna dair olan örneği yapalım
notes[0] = 99
notes

##########################################################
# Liste Methodları (List Methods)
##########################################################
dir(notes)

#################################################
# len: builtin python fonksiyonu, boyut bilgisi.
###################################################

len(notes)
len(not_nam)

######################################################
# append: eleman ekler
###################################################

notes.append(100)
notes

############################
# pop: indexe göre siler
############################

notes.pop(0)

##############################
# insert: indexe ekler
##############################

notes.insert(2, 99)  # 2. indexe 99'u gir

######################################################
# Sözlük (Dictionary)
#####################################################
# Değiştirilebilir.
# Sırasız. (3.7 sürümünden sonra sıralı.)
# Kapsayıcı   Yani içerisinde birden fazla veri yapısını aynı anda tutabilir
# key-value

dictionary1 = {"REG": "Regression",
               "Log": "Logistic Regression",
               "CART": "Classification and Reg"}

dictionary1["REG"]

dictionary2 = {"REG": ["RMSE", 10],
               "LOG": ["MSE", 20],
               "CART": ["SSE", 30]}

dictionary3 = {"REG": 10,
               "LOG": 20,
               "CART": 30}

dictionary2["CART"][1]

###################################
# Key Sorgulama
###############################

"YSA" in dictionary1
dictionary3.get("REG")

################################
# Value Değiştirmek
#############################

dictionary3["REG"] = ["YSA", 10]
dictionary3

#############################
# Tüm Key'lere Erişmek
############################

dictionary3.keys()
dictionary3.values()

##############################################
# Tüm Çiftleri Tuple Halinde Listeye Çevirme
###############################################

dictionary3.items()

##################################################
# Key-Value Değerini Güncellemek
##################################################

dictionary3.update({"REG": 11})

#################################################
# Yeni Key-Value Eklemek
##################################################

dictionary3.update({"RF": 10})
dictionary3

######################################################
# Demet (Tuple)
######################################################
# Listelerin değişime kapalı kardeşleridir de diyebiliriz.
# Değiştirilemez
# Sıralıdır. Yani elemanlarına erişilebiliyor, index seçim imkanı sağlıyor.
# Kapsayıcıdır.

t = ("John", "mark", 1, 2)
type(t)
t[0]
t[0:3]

t[0] = 99  # hata verir çünkü değiştirilemez.
# değiştirmek istiyorsak listeye çevirip istediğimiz değişimi yapıp sonra tekrar tuple çevirebiliriz fakat tuple
# güvenli çalışmayı sağlar yani değiştirilmesini istemediğimiz durumlarda kullanabiliriz.

##########################################################
# Set (Küme)
#########################################################
# Değiştirilebilir
# Sırasız + Eşsizdir.
# Kapsayıcıdırlar
# Hız gerektiren ve küme işlemlerinin yapılması(kapsayan,kesişim gibi) gerektiğinde kullanılır.

######################################
# difference(): İki kümenin farkı
######################################

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])
type(set1)

# set1'de olup set2'de olmayanlar. operatör kullanılmış halini de yapalım.
set1.difference(set2)
set1 - set2
# set2'de olup set1'de olmayanlar.operatör kullanılmış halini de yapalım.
set2.difference(set1)
set2 - set1
#####################################################################
# symmetric_difference(): İki kümede de birbirlerine göre olmayanlar
#####################################################################

set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

###########################################################
# intersection(): İki kümenin kesişimi
##########################################################

set1.intersection(set2)
set2.intersection(set1)

# aynısını operatörlerle de yapabiliriz
set1 & set2

#########################################################
# union(): İki kümenin birleşimi
#########################################################

set1.union(set2)
set2.union(set1)

######################################################
# isdisjoint(): İki kümenin kesişimi boş mu?
#####################################################
# genelde komutun önünde is varsa true false cevabı verir

set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])

set1.isdisjoint(set2)
set2.isdisjoint(set1)

######################################################
# issubset(): Bir küme diğer kümenin alt kümesi mi?
#######################################################

set1.issubset(set2)  # set1, set2'nin alt kümesimidir. TRUE
set2.issubset(set1)  # set2, set1'in alt kümesimidir. FALSE

#################################################################
# issuperset(): Bir küme diğer kümeyi kapsıyor mu?
#################################################################

set2.issuperset(set1)  # set2, set1'i kapsıyor mu?. True
set1.issuperset(set2)  # set1, set2'yi kapsıyor mu?. False

########################################################################################################################
#                                                                                                                      #
#                                            FONKSİYONLAR                                                              #
#                                                                                                                      #
########################################################################################################################

###############################
# Fonksiyon Okuryazarlığı
################################

# Kodlarımızı alttaki Python console kısmından da yazabiliriz böylece yazdığımızı programı etkilemiş olmaz.

#                               ***** Ortamlarda Satan Bilgi ******
# Parametre fonksiyon tanımlanması aşamasında ifade edilen değişkenlerdir(sep, end gibi), argüman ise bu fonksiyonlar,
# çağrıldığında bu parametre değerlerine karşılık girilen değerlerdir. Fakat yaygın kullanımı bunların hepsine argüman
# deme eğilimi yönündedir.

##########################
# sep
##########################
# defualt olarak boşluk bırakır, boşluk yerine isteğimiz bir değer ile biçimlendirebiliriz.

print("a", "b")
print("a", "b", sep="__")


##################################################################3
# help(print) yada ?print şeklinde dokümantasyonlara ulaşabiliriz
###################################################################

########################################
# Fonksiyon Tanımlama
########################################

def calculate(x):
    print(x * 2)


calculate(5)


# iki argümanlı/parametreli bir fonksiyon tanımlayalım.

def summer(arg1, arg2):
    print(arg1 + arg2)


summer(13, 12)  # argüman sırası önemli
summer(arg2=13, arg1=12)


#####################################################################################################
# Docstring: Fonksiyonlarımıza herkesin anlayabileceği ortak bir dil ile bilgi notu ekleme yoludur.
#####################################################################################################

def summer(arg1, arg2):  # 3kere çift tırnak tuşuna basıp entera basıyoruz ve altaki yeşil renkteki kısım çıkıyor.
    # Birden fazla yolu vardır bizim yaptığımız numpy yolu ile. Ayarlardan numpy, google gibi seçenekleri değiştirebiliriz.
    """

    Parameters/Args
    ----------
    arg1: int, float
        Buraya da görevini girebiliriz
    arg2: int, float

    Returns
        int, float
        
    Examples:
        örneklerini de koyabiliriz.
        
    Notes: 
        notlarda koyabiliriz
    -------
    """  # help menüsünden bunu incelediğimizde bilgi notu gelmiş olacaktır.ya da fonk üzerine gelmemiz de yeterli olur.

    print(arg1 + arg2)


summer(1, 3)


#############################################
# Fonksiyonlarda Statement/Body Bölümü
############################################

# def function_name(parameters/arguments):   # bir fonksiyon parametresiz argümansız da olabilir
#     statements (function body)

def say_hi():
    print("Merhaba")
    print("Hi")
    print("Hello")


say_hi()  # fonksiyonu çağırmamız gerekiyor


def say_hi(string):
    print(string)
    print("Hi")
    print("Hello")


say_hi("miuul")

"""
def multiplication(a, b):
    c = a * print(c)


multiplication(10, 9)
"""
# Girilen değerleri bir liste içinde saklayacak fonksiyon yazalım.

list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(
        c)  # kullandığımız bazı metotlar yeniden atama işlemi yapmaya gerek kalmaksızın ilgili veri yapısında kalıcı bir değişiklik meydana getirebilir. append de onlardan biridir.
    print(list_store)


add_element(1, 8)
add_element(2, 10)
add_element(180, 20)

#                           *** Ek Bilgi ***
#    c = a* b
#    list_store.append(c)
#    print(list_store)
# Üsteki kod kısmı local scoptur yani lokal etki alanıdır. Global etki alanında (sağ en alttaki kısım) gözükmez.
# Geçici bir şekilde kullan at olarak oluşturulup atılıyor. list_store global etki alanında onu görebiliyoruz.

##########################################################################
# Ön Tanımlı Argümanlar / Parametreler (Default Parameters / Arguments)
#########################################################################
""" 
Argüman, parametre ifadeleri birbiri yerine kullanılabilmektedir. Teknik olarak parametre fonksiyon tanımlanması 
esnasında kullanılan ifadelerdir. Bu parametreler fonksiyonun çağrılması esnasında değerlerini aldığında argüman olarak
alınır. Fakat yaygın kullanım argüman ifadesi hepsi için kullanılmaktadır.
"""


def divide(a, b):
    print(a / b)


divide(1, 2)


def divide(a, b=1):
    print(a / b)


divide(1)


def say_hi(string="Merhaba"):  # Kullanıcı hiçbirşey girmezse Merhaba yazdırmış olucaz.
    print(string)
    print("Hi")
    print("Hello")


say_hi("mrb")

#################################################################
# Ne Zaman Fonksiyon Yazma İhtiyacımız Olur?
################################################################
"""
Bir belediyede çalıştığımızı varsayalım ve akıllı sokak lambalarından gelen sinyaler var ısı, nem ve pil durumu gibi bu
gelen veriler ile işlem yapmamız gerektiğini düşünelim.
"""


# DRY felsefesi vardır açılımı ise don't repeat yourself yani tekrar eden bir durum varsa fonksiyonları kullanmalıyız

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)


calculate(98, 12, 78)
calculate(105, 32, 28)
calculate(65, 14, 88)


########################################################################
# Return: Fonksiyonun Çıktılarını Girdi Olarak Kullanmak
########################################################################

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)


calculate(98, 12, 78) * 10  # yazdığımızda hata alırız çünkü nonetype bir değerle int'i çarpamayız


def calculate(varm, moisture, charge):
    return (varm + moisture) / charge


calculate(98, 12, 78) * 10
a = calculate(98, 12, 78)


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge

    return varm, moisture, charge, output


calculate(98, 12, 78)
type(calculate(98, 12, 78))

varm, moisture, charge, output = calculate(98, 12, 78)


###############################################
# Fonksiyon İçerisinden Fonksiyon Çağırmak
###############################################

def calculate(varm, moisture, charge):
    return int(varm + moisture) / charge


calculate(90, 12, 12) * 10


def standardization(a, p):
    return a * 10 / 100 * p * p


standardization(45, 1)


def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)


all_calculation(1, 3, 5, 12)


def all_calculation(varm, moisture, charge, a, p):
    print(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)


all_calculation(1, 3, 5, 19, 12)

############################################################################
# Lokal ve Global Değişkenler
###########################################################################

list_store = [1, 2]


def add_elements(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(1, 9)

# lokal etki alanından global etki alanını etkilemiş olduk c'yi list içinde yazdırarak


########################################################################################################################
#                                                                                                                      #
#                                              Koşullar & Döngüler                                                     #
#                                                                                                                      #
########################################################################################################################

##################################
# Koşullar (Conditions)
##################################

# True-False'u hatırlayalım
1 == 1
1 == 2

# if: eğer demektir. Bir sorgu var ve benim için yanıtı önemlidir der if.
if 1 == 1:
    print("evet")

if 1 == 2:  # Çıktı alamayız çünkü if ancak yazdığımız işlemin sonucu true olursa alt bloktaki işlemi yapar.
    print("Hayır")

number = 11

if number == 10:
    print("number is 10")

number = 10

if number == 10:
    print("number is 10")


# Az önce yaptığımız örnekte DRY'ye takıldık. Numara kontrolü yapan bir fonksiyon yazalım.

def number_check(number):
    if number == 10:
        print("Number is 10")


number_check(12)
number_check(10)


################################################
# else
###############################################

# if koşulu çalışmadığında ne yapmasını istediğimizi yazalım.

def number_check(number):
    if number == 10:
        print("Number is 10")
    else:
        print("Number is not 10")


number_check(12)


#############################
# elif
#############################

def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")


number_check(10)

#################################################
# Döngüler (Loops)
#################################################
"""
Döngü ifadesi bize üzerinde itarasyon yapılabilen nesneler üzerinde, gezinmeyi ve bu gezinmeler sonucunda yakalamış 
olacağımız her bir elemanın üzerinde çeşitli işlemler yapabilme imkanı sağlar.
"""
# for loop

students = ["John", "Mark", "Venessa", "Mariam"]

# for bu_elemanları_temsil_ettiğin_şey in döngüyü_gerçekleştireceğin_yer:

for student in students:
    print(student)

students = ["John", "Mark", "Venessa", "Mariam"]
for student in students:
    print(student.upper())

# Alttaki kodlar doğru farklı compilerlarda çalıştı nedense burda çalışmadı
salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary * 20 / 100 + salary))


# rate: zam miktarı

def new_salary(salary, rate):
    return int(salary * rate / 100 + salary)


new_salary(1500, 10)

for salary in salaries:
    print(new_salary(salary, 10))


salaries2 = [10700, 25000, 30400, 40300, 50200]

for salary in salaries2:
    print(new_salary(salary, 15))

# maaşı 3000'nin altında ve üstünde olanlara farklı zamlar yapalım.

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 10))

