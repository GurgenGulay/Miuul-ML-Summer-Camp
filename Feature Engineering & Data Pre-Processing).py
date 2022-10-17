"""
If your Data is bad, your machine learning tools are useless
by Thomas C. Redman
"""
"""
Applied machine learning is basically feature engineering
Andrew Ng
"""
# Özellik Mühendisliği: Özellikler üzerinden gerçekleştirilen çalışmalar. Ham veriden değişken üretmek.
# Veri ön işleme: Çalışmalar öncesi verinin uygun hale getirilmesi sürecidir.

########################################################################################################################
#                                                                                                                      #
#                                           Outliers                                                                   #
#                                                                                                                      #
########################################################################################################################


########################################
# Outliers ( Aykırı Değerler )
########################################
"""
Veride ki genel eğilimin oldukça dışına çıkan değerlere aykırı değerler denir.
Aykırı değerler için kullanılacak yöntemler;
 1. Sektör Bilgisi / Emlakçılar metre kareye göre gibi...
 2. Standart Sapma Yaklaşımı / ort=10 std.sap.=5  5'in altında ve 15'in üstünde olan değerler aykırı değer olarak yorumlanabilir.
 3. Z-Skoru Yaklaşımı  / ilgili değişken standartlaştırılır ve ort 0 olur  ve sağından(+), solunda(-) değerler eşik değeri olur. 
 4. Boxplot (interquartile range - IQR) Yöntemi / tek değişkenli olarak yaygınca tercih edilendir.
Çok değişkenli olarak ise Lof yöntemini kullanıcaz.

Aykırı değer hesaplamalarında kritik nokta eşik değerini belirlemektir.
"""

# Öncelikle özellik mühendisliği bölümünde lazım olacak kütüphaneleri kuralım.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# Görsel ayarlamalar
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data


dff = load_application_train()
dff.head()


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()

#######################################################################
# Aykırı Değerleri Yakalama
######################################################################

sns.boxplot(x=df["Age"])  # aykırı değerleri kutu grafiğinde görelim
plt.show()

#######################################
# Aykırı Değerler Nasıl Yakalanır?
######################################

q1 = df["Age"].quantile(0.25)  # %25'lik çeyreği getir diyoruz.
q3 = df["Age"].quantile(0.75)  # ½75'lik olarak hesaplamamız lazım

iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr  # alt eşik değerimizi de bulduk ama yaş - olamayacağı için bunu görmezden gelicez

df[(df["Age"] < low) | (df["Age"] > up)]  # belirlediğimiz sınır değerlere göre aykırı olan değerleri gösterelim

df[(df["Age"] < low) | (df["Age"] > up)].index  # eğer indexlerine ihtiyacımız olursa

#######################################
# Aykırı Değer var mı yok mu?
######################################

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)  # true döndü aykırı değerler var dedi

# (~)tilda koyarak bir nevi değilini alıyoruz yani aykırı olmayanları getir diyoruz.

df[~((df["Age"] < low) | (df["Age"] > up))].any(axis=None)

df[(df["Age"] < low)].any(
    axis=None)  # alt eşik değerimizin altında değer var mı yok mu baktık false döndü doğal olarak -'li yaşlarda yokmuş.


#######################################
# İşlemleri Fonksiyonlaştırmak
######################################


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


outlier_thresholds(df, "Age")
low, up = outlier_thresholds(df, "Age")

outlier_thresholds(df, "Fare")
low, up = outlier_thresholds(df, "Fare")

df[(df["Fare"] < low) | (df["Fare"] > up)].head()

df[(df["Fare"] < low) | (df["Fare"] > up)].index


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(
            axis=None):  # eğer bu şekilde bool cevaplı şey istiyorsak if, else'in sonucuda bool olmalıdır.
        return True
    else:
        return False


check_outlier(df, "Age")
check_outlier(df, "Fare")

##################################
# grab_col_names
###################################
"""
Öyle bir işlem yapmalıyım ki bana otomatik olarak sayısal değişkenleri, kategorik değişkenleri, kategorik olmasa bile 
aslında kategorik olan işlemleri, kategorik olmasada aslında kategorik olan işlemleri getirmiş olsun.
"""
dff = load_application_train()
dff.head()

"""
grab_col_names
hayat kurtaran fonksiyondur çünkü bu fonk. sayesinde kategorik,numerik ve numerik görünümlü kategorik değişkenlere ulaşabileceğiz.
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
:parameter
 -------
    dataframe: dataframe
            Değişken isimleri alınmak istenen dataframe
    cat_th: int, optional
            numerik fakat kategorik olan değişkenler için sınıf eşik değeri 
    car_th: int, optinal
            kategorik fakat kordinal değişkenler için sınıf eşik değeri
            
:returns
---------
    cat_cols: list
            Kategorik değişken listesi
    num_cols: list
            Numerik değişken listesi
            
"""


# katogik d. = cinsiyet, Embarked / sayısal görünümlü kategorik d. = Pclass, Survived, SibSp / kategorik görünümlü olup
# bilgi taşımayan d. = Name, Ticket, Cabin

def grab_col_names(dataframe, cat_th=10,
                   car_th=20):  # bu kısımlar yorum veri setine göre değişebilir.Bizim yaptığımız ise şöyledir;
    # Eğer bir değişkenin içerisindeki sayısal değişkenin sınıf sayısı 10'dan azsa bu aslında sayısal görünümlü bir kategorik değişken olabilir.
    # Eğer bir kategorik değişkenin sınıf sayısı 20'den fazlaysa bu aslında sayısal görünümlü bir kardinal değişken olabilir.

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtype != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtype == "O"]  # kategorik ama kardinal değişkenler

    cat_cols = cat_cols + num_but_cat  # cat_cols listemizi baştan düzenliyoruz

    cat_cols = [col for col in cat_cols if
                col not in cat_but_car]  # kategorik ama kardinal değişkenlerde olmayanları seç

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_cols = [col for col in num_cols if
                col not in num_but_cat]  # saklananlar vardı numeric olarak gözüküp kategorikler vardı bunun
    # içinde num_but_cat listesinden num_cols lisesinde sorguluyorum num_but_cat bunun içinde olanları alma diyorum.

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

    cat_cols, num_cols, cat_but_car = grab_col_names(df)


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col))


#############################################
# Aykırı Değerlerin Kendilerine Erişmek
#############################################
# shape 0 alırsak gözlemi verir

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


grab_outliers(df, "Age")
grab_outliers(df, "Age", True)
age_index = grab_outliers(df, "Age", True)

###########################################################
# Aykırı Değer Problemini Çözme
###########################################################

######################
# Silme
######################

low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outlier = dataframe[~((df["Fare"] < low) | (df["Fare"] > up))]
    return df_without_outlier


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

# Hücrede ki bir tane aykırılıktan dolayı silme işlemi yaptığımızda diğer tam olan gözlemlerdeki verilerden de oluyoruz
# bundan dolayı bazı senaryolarda silmek yerine bu değerleri baskılama yöntemiyle baskılamayı da tercih edebiliriz.

#########################################################
# Baskılama Yöntemi (re-assignment with thresholds)
#########################################################
"""
Kabul edilebilir bazı eşik değerlerimiz vardı bu eşik değerlerin üzerinde kalan değerler eşik değerleriyle değiştirilir.
Silme yönteminde ortaya çıkabilecek veri kaybetme ihtimalini istemediğimizde baskılama yöntemini kullanabiliriz.
"""

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]

# aynısını loc ile de yapabiliriz. aykırı değerleri getir diyoruz.

df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

# up limitleri artık aykırıların yeni değerleri olarak değiştirdik

df.loc[(df["Fare"] > up), "Fare"] = up

# alt sınırdakilere de yapalım ama bizde alt sınırda aykırı değer olmadığı için birşey değiştirmeyecektir.

df.loc[(df["Fare"] < low), "Fare"] = low


# yaptıklarımızı fonksiyonlaştıralım

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))  # aykırı değer var mı diye sorduk var dedi

for col in num_cols:
    replace_with_thresholds(df,
                            col)  # bir üstte yaptığımız fonksiyonu kullandık ve baskılama yaptık ve tekrar sorduk üstten bu sefer yok dedi

###################
# Recap ( Özet )
####################
# Saptama işlemleri

df = load()
outlier_thresholds(df, "Age")  # aykırı değeri saplama işlemi
check_outlier(df, "Age")  # bu threshold'lara göre outlier var mı yok mu sorusunu sorduk
grab_outliers(df, "Age", index=True)  # outlier'ları bize getir dedik

# Tedavi işlemleri :)

remove_outlier(df, "Age").shape  # silme yöntemini kullandık ama atama yapmadık o yüzden kalıcı değişiklik sağlamadı.
replace_with_thresholds(df,
                        "Age")  # baskılama yöntemini kullandık. İçeride kullandığımız loc yapısından dolayı kalıcı değişiklik sağlar.
check_outlier(df,
              "Age")  # bu threshold'lara göre outlier var mı yok mu sorusunu tekrar sordup işlemlerimizin sonucunu görmüş olduk.

# en önemli fonksiyonlar nedir diye soracak olursak outlier_thresholds() ve replace_with_thresholds() diyebiliriz.
# grab_col_names() 'in önemini unutmamak gerekir.

########################################################################
# Çok Değişkenli Aykırı Değer Analizi ( Local Outlier Factor )
########################################################################
"""
Tek başına aykırı olamıyacak değerler birlikte ele alındığında aykırılık yaratıyor olabilir.
Örneğin 3kere evlenmiş olmak aykırı değildir aynı zamanda 18 yaşında evlilikte aykırı değildir ama 18 yaşında 3 kere 
evlenmiş olmak aykırı değer olabilir.
"""

# ----Lof Yöntemi Ne Yapar?
"""
Gözlemleri bulundukları konumda yoğunluk tabanlı skorlayarak buna göre aykırı değer tanımı yapabilmemizi sağlar.
----Peki bu ne demek?----
Bir noktanın lokal yoğunluğu demek ilgili noktanın etrafındaki komşuluklar demektir. Eğer bir nokta komşularının 
yoğunluğundan anlamlı bir şekilde düşük ise bu durumda bu nokta daha seyrek bir bölgededir yani demek ki bu aykırı
değer olabilir yorumu yapılır.
Resimdeki A aykırı değerdir
"""
###############################
# Inliers ve outliers
#############################
"""
Lof yöntemi der ki ben size bir scor vericem bu benden aldığınız scor 1'e ne kadar yakın olursa o kadar iyidir der.
Dolayısıyla 1'den uzaklaştıkça ilgili gözlemin outlier olma ihtimali artar(küçük notalar inlier büyükler ise outlier
olabileceğine dair fikir verir.)
------ Bu değerlere müdahale edebilir miyiz?
Evet threshould değerini değiştirip inlier sayısını artırabiliriz.
"""
########################
# Mülakat Sorusu
########################
# Elimizde 2'den fazla değişken olabilir buna rağmen 2 boyutlu gösterim yapmak mümkündür

# Mülakat Sorusu: Elimizde 100 tane değişen var nasıl 2 boyutta görselleştirme yapabilirim?
"""
Eğer elimdeki 100 değişkeni o 100 değişkenin taşıdığı bilginin büyük bir miktarını taşıdığı varsayabileceğimiz 2 boyuta
indirgeyebilirsem yapabilirim. Bunu da "Temel Bileşen Analizi Yöntemi" ile gerçekleştirebiliriz.
"""

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_thresholds(df, "carat")
df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_thresholds(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape

# outlier_thresholds literatürde 25,75'liktir(shape). Eğer 25,75'lik silseydik ciddi bir veri kaybı olcaktı yada
# doldursaydık da tortu yaratacaktı yani kendi kendimize veri setine gürültü ekleyecektik. Yani durduk yere problem oluşacaktı.
# Ağaç yöntemleri kullanıyorsak bu sebeple hiç dokunmamayı tercih etmeliyiz yada Vahit Keskin'şn yaklaşımı olan ucundan
# traşlama, ekstra ekstra aykırı olanları çıkarma yaklaşımını sergilememiz gerekecektir. Burada da bunun ne kadar ciddi
# bir problem olduğunu ve gözden kaçmaması gerektiğini ispatlayacağız.

# Tek başına baktığımızda çok yüksek sayıda aykırılıklar geldi. Birde buna çok değişkenli yaklaştığımızda ne olacak.
# Daha önce yukarıda import ettiğimiz LocalOutlierFactor methodu getiriyoruz
"""
Dipnot: Buradaki komşuluk sayısı değişebilir 5 gibi 3 gibi istediğiniz sayıyı 
yapabilirsiniz.Ama buradaki problem denemeler sonucunda elde ettiğimiz komşuluk sayılarının hangisinin daha iyi olacağını
yorumlayamıyor olacak olmamızdır dolayısıyla burada LocalOutlierFactor methodunda ön tanımlı değer olan 20'yi kullanmak 
tercih edilmelidir.
"""

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)  # çıktı olarak local outlier scorlarını getiriyor olacak

df_scores = clf.negative_outlier_factor_  # bu şekilde scorları tutuyorum
df_scores[
0:5]  # - değerlerle verdi ona göre değerlendiricez ama eğer - değerler istemezsekde aşağıdaki gibi yazabiliriz
# df_scores = -df_scores

# Ama biz fonksiyonun bize verdiği negatif değerlerle kullanmayı tercih edicez bunun sebebi eşik değere karar vermek için
# kullanıcı olarak bir bakış gerçekleştirmek istediğimizde oluşturacak olduğumuz elbow(dirsek) yöntemi grafik tekniği ile
# daha rahat okunabilirlik açısından - olarak bırakıcaz.


# Buradaki değerlerin 1'e yakın olması inlier olması durumunu gösteriyor demiştik şimdi artık -1'e yakın olması inlier
# olması durumunu gösteriyor gibi değerlendireceğiz.


np.sort(df_scores)[0:5]  # bu değerleri küçükten büyüğe sıralayalım ve en kötü 5 değeri görelim

# Eşik değeri belirlemeliyiz.İstediğimiz herhangi biryeri belirleyebiliriz. Kullanıcının burada müdahale edebiliyor olması
# bir avantajdır. Eğer müdahale etmek istemiyorum programatik yapmak istiyorum gibi bir yorum varsa;


# Temel bileşen analizinde dirsek PCA'de kullanılan elbow yöntemi var. Bu yönteme göre bunu belirleyebiliriz.

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

# Grafikteki herbir nokta eşik değerleri temsil ediyor ve bu eşik değerlere göre grafik oluşturulmuş.
# Nereden bölmem gerekiyor?
# Elimde bazı gözlemler(x) ve bunların scorları(y) var ve bu scorlara karşılık nereden bölmem gerektiği problemi var,
# diyorumki en marjinal değişiklik yani en dik eğim değişikliğinin bariz olduğu noktayı eşik değer olarak belirleyebilirim.

th = np.sort(df_scores)[
    3]  # eşik değeri seçtik 3.indexteki ve bundan daha düşük olanları aykırı değer olarak belirleyeceğim

df[df_scores < th]

df[df_scores < th].shape  # 3 tane olduğunu gördük

# Gördüğümüz gibi daha önce tek değişkene baktığımızda binlerce varken birden fazla değişkene baktığımızda 3 aykırı değere ulaştık.

#######################################
# Acaba bu değerler niye aykırı?
#########################################

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

# Tek tek bu 3 aykırı değeri kıyasladığımızda ortalamalara yakın tek başına aykırı özellikleri var ama çok değişkene takılma
# sebebibi derinliği ve carat bu ölçüdeyken fiyatının ... olması sebep olmuş olabilir gibi yorumlar yapılabilir.

# index bilgilerini yakalayalım

df[df_scores < th].index

# bu 3 aykırı değeri silelim

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# Eğer baskılama yöntemi kullanabilir miyiz diye soracak olursak kullanabiliriz ama şöyle bir problem var, kimle baskılayacağız
# bir tane rastgele gözlem seçebiliriz yada en ortalarda olan gözlemi seçebiliriz bu gözlemle aykırılığı değiştirebiliriz.
# Bu senaryo için zaten gözlem sayıyı az olduğu için bir problem olmayacaktır ama yüzlerce aykırı değer hesaplanmış olsa
# thresholds'da göre bu şu olucak o zaman yüzlerce aynı değeri verisetine baskılamış, değiştirmiş olucaz.
# Şimdi artık hücreler ile değilde gözlem birimleri ile ilgilendiğimizden bir gözlem birimine baskılama yapmamız gerektiğinden
# aykırılığı barındıran gözlemi tamamen kaldırıp yerine başka bir gözlem koymamız lazım buradaki problemde şu ki çoklama
# kayıt üreticez kendi elimizle, zorla veriyi bozucaz tortu oluşturucaz. Dolayısıyla gözlem sayısı bir miktar fazlaysa buraya
# değiştirmeyle ve benzeri noktalarla dokunmak ciddi problemlere sebep olabilecektir. O zaman ne yapıcaz diye soracak olursak
# eğer ağaç yöntemleriyle çalışıyor isek bunlara hiç dokunmayacağız en kötü ucundan traşlama dediğimiz en en aykırı olan
# outlier_thresholds ve replace_with_thresholds kullanarak gözlemlere kendi içinde yaklaşıcaz.

########################################################################################################################
#                                                                                                                      #
#                                     Missing Values ( Eksik Değerler )                                                #
#                                                                                                                      #
########################################################################################################################

"""
-Gözlemlerde eksiklik olması durumunu ifade etmektedir.
Eksik veri problemi nasıl çözülür?
 3 yöntem vardır.
    - Silme
    - Değer Atama Yöntemleri (mod,medyan,ortalam gibi)
    - Tahmine Dayalı Yöntemler
    
*** Eksik veri ile çalışırken göz önünde bulundurulması gereken önemli konulardan birisi: Eksik verinin rassalığı
    yani eksik verinin rastgele ortaya çıkıp çıkmadığı.
    
"Eksik değere sahip gözlemlerin veri setinden direkt çıkarılması ve rassalığın incelenmemesi, yapılacak istatistiksel 
çıkarımların ve modelleme çalışmalarının güvenilirliğini düşürecektir.(Alpar, 2011)"


Diyelim ki bir değişkeni inceliyoruz değişkenimiz kredi kartı harcama değişkeni olsun örneğin bu aylık ort. harcama yada
kredi kartı harcama değişkeninde bazı değerlerin NAN olduğunu düşünelim. Eğer bu NAN'ler rastgele ortaya çıktıysa problem
yok istediğimiz yöntemi kullanabiliriz.

Aynı senaryo için bu sefer kredi kartı var mı yok mu değişkeni olsun . Bir kişinin kredi kartı yoksa yani 0'sa zaten kredi
kartı harcaması 0 olacaktır. 0 nümerik olarak basılmamış olacağından dolayıda 0 olması yerine NAN olacaktır. Böylece
eksikliğimiz rastgele değil çünkü kredi kartı olup olmama durumu ifade eden başka bir değişkene bağlı işte buna bir
değişkendeki eksiklik başka bir değişken etkisinde çıkmıştır denir.Bundan dolayı kredi kartı harcaması değişkeni
üzerindeki eksik veriler rastgele değildir.Adım atarken dikkatli olmamız gerekir ve bu yapısallığın nerden kaynaklandığını
bulup belki çözmekle uğraşmak gerekir.
"""
df = load()
df.head()

# Eksik gözlem var mı yok mu sorgusu
# isnull() methodu df'e tüm hücreleri dolaş bak bakalım eksik veri var mı diye sorar,gelen T yada F yanıtlarını values
# ile tutarız ve any ile eğer bir tane bile True varsa bunu getir deriz.

df.isnull().values.any()

# değişkenlerdeki eksik değer sayısı
df.isnull().sum()

# değişkenlerdeki tam değer sayısı (dolu olan)
df.notnull().sum()

# veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()

# neden bu kadar yüksek bu sayı diyecek olursak çünkü bir satırda en az bir tane bile bir hücrede bile eksiklik varsa bu
# durumda o eksikliğide sayacaktır.Bu yüzden kendisinde en az 1 tane eksik hücre olan satır sayısıdır bu sayı.

# en az bir tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

# eksik değer oranını bulmak için
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# sadece eksik değere sahip değişenlerin isimlerini yakalamak için
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)

#####################################
# Eksik Değer Problemini Çözme
####################################
"""
Eksik değer ve aykırı değer problemleri genel hatları itibariyle çok iyi hakim olmak lazım ama nerede ne kadar etkileri 
olduğunu da iyi bilmek lazım. Doğrusal yöntemlerde ve Gradient Descent temelli yöntemlerde bizim için bu teknikler çok 
daha hasas iken ağaca dayalı yöntemlerde bunların etkisi çok daha düşüktür.
"""
missing_values_table(df)

###############################
# Çözüm 1: Hızlıcı Silmek
############################

df.dropna().shape  # gözlem sayısı çok azaldı çünkü 1 satırda en az 1 tane bile Nan değer varsa dropna onları da silmiş olacaktır.

#########################################################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
########################################################

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# Elimde birçok değişken olursa ne yapabilirim. Not: axis=0 satırları axis=1 sutünları ifade eder.

# df.apply(lambda x: x.fillna(x.mean()), axis=0) # hata aldık çünkü veri setinde hem kategorik tipde hem de sayısal tipte değişkenler var

# diyoriz ki: eğer ilgili değişkenin tipi objectten farklıysa bu değişkeni ort. ile doldur eğer objectten farklı değilse olduğu gibi kalsın.
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

df2 = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

df2.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).head()

####################################################
# Kategorik Değişken Kırılımında Değer Atama
######################################################

df.groupby("Sex")["Age"].mean()

df["Age"].mean()

# cinsiyete göre kırmak, çıkan ortalamayı yani erkeklere göre ve kadınlara göre eksiklik varsa bunlara farklı bir değer atasak daha doğru olur.

# yaş değişkenini doldur(fillna) diyorum.Neye göre doldur cinsiyete göre veri setini grouply'a al daha sonra yaş değişkenini seç.
# Bu yaş değişkeninin ortalamasını aynı groupby kırılımında ilgili yerlere yaz.

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# yukarıdaki işlemin daha açık yazılmış formunu yapalım

# yaş değişkeninde cinsiyeti kadın olup Nan olanları getir.Biz ne yapmaya çalışıyorduk groupby kırılımında uygun değerleri uygun
# yerlere atamaya çalışıyorduk
df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

# Gördüğümüz gibi Age değişkeninde eksik veri kalmadı.
df.isnull().sum()

# yukarıda yaptıklarımızı age gibi diğer kategorik değişkenlerde de yapabiliriz.

#########################################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurmak
########################################################
"""
Bir makine öğrenmesi yöntemiyle tahmine dayalı bir şekilde modelleme işlemi gerçekleştireceğiz.
Eksikliğe sahip olan değişkene bağımlı değişken diğer değişkenleri bağımsız değişkenler gibi kabul edip bir modelleme 
işlemi gerçekleştireceğiz ve bu modelleme işlemine göre eksik değerlere sahip olan noktaları tahmin etmeye çalışıcaz.
Fakat burada birkaç kritik konu olacak. Bunlardan;
    1. Kategorik değişkenleri one hot encodera sokmamız lazım yani bir modelleme tekniği kullanıcak olduğumuzdan dolayı
bu modelin, değişkenlerin bizden beklediği bir standart var bundan dolayı bu standarta uymamız gerekmekte.
    2. Uzaklık temelli bir algoritma olduğundan dolayı değişkenleri standartlaştırmamız lazım.
"""

# drop_firs=True yaptığımız değişkenin ilk sınıfını atıcak 2.sini tutacak yani böylece elinde örneğin elinde cinsiyet
# gibi male,female kategorik bir değişken olduğunda bu kategorik değişkeni de binary bir şekilde temsil ediyor olucak.

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

# Amacımız: Kategorik işlemleri iki sınıf yada daha fazla sayıda sınıfa sahip olan değişkenleri nümerik bir şekilde ifade etmek.
# iki listeyi toplamışız, get_dummies methodu bütün değişkenleri birlikte versek bile sadece kategorik değişkenlere bir dönüşüm uygulamaktadır.
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

# değişkenlerin standarttılaştırılması

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff),
                   columns=dff.columns)  # veri setine uygulayıp istediğimiz formatta olmayacağı
# için bir dataframe geri(pd.DataFrame ile) çevirip df isimlerini dff.columns alıyoruz.
dff.head()

# knn'in uygulanması
# knn nasıl çalışır ?
# Özetle bana arkadaşını söyle sana kim olduğunu söyleyeyim der.Örneğin yaş değişkeninde nan olan değerin en yakın dolu
# 5(biz veriyoruz başka bir sayıda verilebilir) komşusuna bakıp ortalamasını alıp nan değere atıyacaktır.
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# Evet doldurduk ve işlem bitti fakat biz bu doldurğumuz yerleri görmek istiyor isek

# standartlaştırma işlemini geri alalım
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull(), ["Age",
                            "age_imputed_knn"]]  # hangi indexte ki nan değere, knn ile yerine ne konduğunu görebiliyoruz.

# Bütün değişkenleri görmek istiyorum acaba nasıl atanmış dersek ise;
df.loc[df["Age"].isnull()]

################################
# Gelişmiş Analizler
#################################
# msno kütüphanesini çağırıp titanik veri setimizi ifade eden df'i uyguluyoruz
# İlgili veri setindeki değişkenlerdeki tam sayıları göstermektedir.
msno.bar(df)
plt.show()

# Değişenlerdeki eksikliklerin birlikte çıkıp çıkmadığıyla ilgili bilgi verir.
msno.matrix(df)
plt.show()

# Eksiklikler üzerine kurulu bir ısı harıtasıdır. Önemli bir haritadır.Eksik değerlerin rassalığı ile ilgileniyoruz ve
# bu harita bize bunun hakkında fikir sahibi olmamıza yardımcı olur.
# +1'e yakınsa pozitif yönlü denir ve pozitif yönlü ilişki olması durumunda değişkenlerdeki eksikliklerin birlikte ortaya
# çıktığı düşünülür. Yani birisinde eksiklik varken diğerinde de vardır. Birinde yokken diğerinde de yoktur gibi
# -1 civarına yakınsa yani negatif bir korelasyon varsa birisinde varken diğerinde yok, birisinde yokken diğerinde var
# gibi ters yönlü ilişki vardır.
# Bizim heatmap tablomuza bakıcak olursak anlamlı değil. Anlamlı olabilmesi için mesela %60'ın üzerinde değerler çıksaydı
# bu eksikliklerin birlikte çıkabileceğinden şüphelenebilirdik.
msno.heatmap(df)
plt.show()

#######################################################################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
#######################################################################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)


# temp_df[col].isnull(), 1, 0 eksilik varsa gördüğün yere yani na olan yerlede 1 yoksa 0 yaz demiş oluruz

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:  # na sahip colonlarda gez
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1,
                                             0)  # örneğin [col(cabin) + string] = çıktısı Cabin_NA_FLAG olur.
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)

# cabin kısmına dikkat edelim. cabin ve çalışan örneğini hatırla!


################################
# Recap ( Özet )
#################################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile doldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine dayalı atama ile doldurma
missing_vs_target(df, "Survived", na_cols)

########################################################################################################################
#                                                                                                                      #
#                                     Encoding Scaling                                                                 #
#                                                                                                                      #
########################################################################################################################
# Değişkenlerin temsil şekillerinin değiştirilmesi.


###############################################
# Label Encoding & Binary Encoding
###############################################
"""
Büyüklük-küçüklük algısı ve ordinallik aranır.
   - Eğer bir kategorik değişkenin 2 sınıfı varsa bu 1, 0 olarak kodlanırsa buna binary encoding denir.
   - Elimizdeki kategorik değişken eğer label encodera sokulmuşsa ve 2'den fazla sınıfı varsa bu durumda label encoding yapılmış olur.
   - Label Encoding > Binary Encoding genel ismi label encodingtir.
Peki neden encode yapmalıyız?
 - Algoritmaların bizden beklediği bir standart format var veriyi buna uygun hale getirmek.
Tek sebebi bu mu?
 - Hayır, tek sebebi bu değildir. Bazen yaptığımız örneğin one hot encoding işlemlerinde amacımız bir kategorik değişkenin
öneli olabilecek sınıfını değişkenleştirerek ona bir değer atfetmek olacaktır.

Dolayısıyla özetle iki çerçeveden encoding işlemlerini gerçekleştiriyoruz.
    1. Kullanıcak olduğumuz modellem tekniklerin bizden beklediği bir standart var
    2. Bu standart ile beraber bizim model tahmin performansımı iyileştirebileceğimiz,geliştirebileceğimiz bazı noktalar
var bunlardan dolayı bu işlemleri yapmak istiyorum.
"""

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]  # alfebatik sıraya göre ilk gördüğüne 0 verir

# Diyelim ki hangisinin 0 hangisinin 1 olduğunu unuttuk ve öğrenmeye ihtiyacımız var;
le.inverse_transform([0, 1])


# fonksiyonlaştıralım
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


"""
Bir problemimiz var.Bir tane değişken olduğunda kolayca uyguladım yaygın problemimiz neydi bunu ölçeklenebilir
yapıyor olmak yani elimde eğer benim yüzlerce değişken varsa nasıl yapıcam.Bu durumda binary col'ları seçebilirim.
İki seçeneğimiz var. Birincisi şuanda gördüğümüz yöntemi uygulayabiliriz, ikincisi one hot encoderı uygulayabiliriz.
One hot encoderı uygularken get_nan... metodunu uygularız ve bu methodu kullanırken drop_first'ü eğer True yaparsak
bu durumda iki sınıflı kategorik değişkenlerde aslında label encoderdan geçirilmiş olur.
"""
"""
Prbleme dönelim elimizde yüzlerce değişken olduğunda ne yapıcam. 2 sınıflı kategorik değişkenşleri seçmenin bir yolunu
bulsam, buluncada bu iki sınıflı değişkenleri label encoderdan geçirsem bu durumda problem çözülür.
"""
df = load()

# in df.columns değişkenin sutünlarında gez / df[col].dtype değişkenin tipine bak / not in [int, float] eğer int yada
# float değilse Dikkat(bir değişkenin tipi integersa ve 2 sınıfı varsa ilgilenmiyorum çünkü zaten binary encode edilmiş.)
# / and df[col].nunique() == 2 ve eşsiz sınıfı 2 olanları seç diyorum
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()  # gördüğümüz gibi cinsiyet artık 1 ve 0'lardan oluşuyor.

# Ek bilgi unique methodunu kullanıp sonra tersini alsaydık değişkenin içerisindeki eksik verileride sınıf olarak görürdü
# nunique metodu eksik veriyi sınıf olarak görmez (number uniqe sayısı 2 olsun dedik).nunique eksik değerleri doldurur.

# Daha büyük bir veri setinde deneyelim

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)

# nunique eksik değerleri doldurur. kanıtı;
df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())

###############################################
# One Hot-Encoding
###############################################
# Değişkenler birbiri üzerinden üretilebilir sanıyor kukla tuzağında. Kukla dediğimiz ayırdığımız değişkenler
# kukla değişken tuzağından kurtulmak için ilk sınıfı drop etmemiz gerekir. drop_first kullanılır.

df = load()
df.head()
df["Embarked"].value_counts()

# get_dummies metodu derki bana dönüştürmek istediğin metodun adını söyle tek onu dönüştüreyim diğerleri aynı kalıcak.
pd.get_dummies(df, columns=["Embarked"]).head()
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()  # çıktıda göründüğü gibi ilk sınıf uçtu.

# Eğer ilgili değişkenlerdeki eksik değerlerde bir sınıf olarak gelsin der isek
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

# get_dummies methodu ile binary encode işlemini yapabiliriz.
pd.get_dummies(df, columns=["Sex"], drop_first=True).head()

# get_dummies methodu ile hem binary hem de one-hot encode işlemini beraber yapabiliriz.
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()


# fonksiyonlaştıralım
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)  # kategorik değişkenleri görmek için çağırdık

ohe_cols = [col for col in df.columns if
            10 >= df[col].nunique() > 2]  # gördüğümüz gibi survived ve cinsiyet değişkenini
# çıkardık işleme sokmak istemedik bazı değişkenleri istemediğimizde bu şekilde kod yazabiliriz.

one_hot_encoder(df, ohe_cols).head()

###############################################
# Rare ( Nadir ) Encoding
###############################################
"""
Genelde model geliştirme süreçlerinde karmaşıklık ile değil basitlik ve genellenebilirlik ile ilgileniyor oluruz. Buradaki
genellenebilirlik ile kastedilen biz herkesi kapsayalım değil, büyük çoğunluğu temsil edelim gibi düşünülebilir. Bu şu
anlama gelir. Bir model kurucaz ve bu modeli sistemlere entegre ediyor olucaz diyelim. Örneğin ev fiyat tahmini, araç 
fiyat tahmini yada buna benzer bir iş dalında bir problemi çözecek bir makine öğrenmesi olsun. Benim yüzbinlerce kaydımın
içinde sadece 2 kere gözükmüş bu değer one-hot encoder'dan geçirdiğimde değişkene dönüşecek. Bu değişkenin sadece 2 tane 
hücresinde 1 diğer örneğin onbinlerce hücresinde 0 yazıyor olucak. Bu bir bilgi taşıyor yada ayırt edici özellik diyebilir 
miyiz? Tabiki diyemeyiz.

Dolayısıyla bu kategorik değişkenin sınıflarını one-hot encoderdan geçirip burda yeni değişkenler oluşturacağım ya bu 
oluşturduğum değişkenlerinde ölçüm kalitesi olsun istiyorum. Bu oluşturacağım değişkenlerinde bağımlı değişkene çeşitli
etkileri olma ihtimalinden gitmek istiyorum bu sebeple ve gereksiz birçok değişken oluşturmak istemediğimizden dolayı
gereksiz değişkenler hem iterasyon süreçlerini hem optimizasyon süreçlerini hem de örneğin ağaç yöntemlerindeki bölünme
süreçlerini,dallanma süreçlerini çok ciddi etkiliyor olacaktır.Bu sebeple gereksiz değişkenlerden uzklaşmak,kurtulmak
çabamızın neticesi olarak Rare Encoding'i kullanabiliriz.

Veri setindeki bir kategorik değişkenin sınıflarındaki az değerler belirli bir eşik değere göre oran yada frekans 
şeklinde olabilir seçilir ve değişken olarak birlikte temsil edilir.Eşik değeri kendimiz belirliyoruz.
"""
#   1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
#   2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
#   3. Rare encoder yazacağız.

#######################################################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
########################################################################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# true yaparsak göreselleştirme olur plot

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

####################################################################################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
####################################################################################

df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


# rare_analyser iki işlemi bir araya getiren fonksiyon
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", cat_cols)


# Elimizde bol miktarda kategorik değişken olan bir veri seti olduğunda %100 rare_analyser fonksiyonunu kullanmamız lazım.
# Hangi kategorik değişkenin sınıfı, hangi frekansa, hangi orana ve bağımlı değişken target açısından nasıl bir etkiye sahip
# bunu mutlaka eğer dokunmasanız dahi bilmemiz gerekmektedir.

###################################
# 3. Rare encoder yazacağız.
#####################################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()  # üzerinde değişiklik yapacağımız için kopyasını aldık

    rare_columns = [col for col in temp_df.columns if temp_df[
        col].dtypes == "O"  # eğer fonk. girilen rare oranından daha düşük sayıda herhangi bir bu kategorik değişkenin sınıf oranı
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(
        axis=None)]  # varsa ve aynı zamanda bu bir kategorik değişkense bunları rare kolon olarak getir diyoruz.

    for var in rare_columns:  # rare kolonlarında gez. var sutün
        tmp = temp_df[var].value_counts() / len(
            temp_df)  # rare sahip kolonların sayısı alınıp toplam gözlem sayısına bölünerek temp_df içerisindeki ilgili rare değişken isimli sınıf oranı hesaplanmış
        rare_labels = tmp[
            tmp < rare_perc].index  # çalışmanın başında verilen orandan daha düşük orana sahip olan sınıflarla verisetine indirge et,indirgedikten sonra kalan indexleri tut(eşik değerden düşük sınıf olan değerler bu indexler)
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[
            var])  # var'ları gördüğün yerlere eğer onlar bu listenin içindeki labellarsa bunların yerine Rare yaz. Değilse olduğu gibi kalsın temp_df[var]

    return temp_df


new_df = rare_encoder(df, 0.01)  # %1'lik eşik değerine göre

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()

#######################################################
# Özellik Ölçeklendirme ( Feature Scaling )
######################################################
"""
  1. Kullanılacak olan yöntemlere değişkenleri gönderirken onlara eşit muamele yapılması gerektiğini bildirmemiz gerekiyor.
Bundan dolayı da standartlaştırmaya ihtiycaımız var.
  2. Özellikle greeden (anlayamadığım bir tane daha isim  söylendi dipsen gibi) algoritamların train, eğitim sürelerini
kısaltmak için
  3. Uzaklık temelli yöntemlerde büyük değerlere sahip değişkenler dominantlık sergilemektedir. Bu da aslında 1. madde ile
aynı anlama sahiptir ama 3. madde gibi bunu da uzaklık temelli yöntemlere yaklaştırarak değerlendirelim. Özellikle KNN,
keynes, PCE gibi uzaklık temelli yada benzerlik-benzemezlik temelli bazı yöntemler kullanılıyor olduğunda burada ölçeklerin 
birbirinden farklı olması durumu yapılacak olan uzaklık, yakınlık, benzerlik, benzemezlik hesaplarında yanlılığa sebep
olmaktadır.

    Dolayısıyla yapacağımız işlem değişkenleri standartlaştırma işlemidir. Bunun sebebi eşit şartlarda yaklaşmak, eğitim
    süresini hızlandırmak ve özellikle uzaklık temelli yöntemlerde yanlılığın önüne geçmektir.

İstisna:    
Ağaca dayalı yöntemlerden birçoğu eksik, aykırı değerlerden, standartlaştırılmalardan etkilenmez. Bunun sebebi dallara 
ayırma işlemi için değişkenlerin değerleri küçükten büyüğe sıralama işlemine tabi tutulur ve bu noktalardan bölünerek
dallanmalar neticesinde entropiler, heterojenlikler, homojenlikler hesaplanır. Bundan dolayı ağaç yöntemleri etkilenmez.
"""
# Yöntemler;

#####################################################
# StandardScaler
######################################################
# Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()

########################################
# RobustScaler
########################################
# Medyanı çıkar iqr'a böl. Klasik standartlaşmaya göre robust aykırı değerlere daha dayanıklıdır, etkilenmiyorda diyebiliriz.

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

########################################
# MinMaxScaler
########################################
# Verilen 2 değer arasında değişken dönüşümü
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

# Karşılaştırma
# num_sarry görselleştirme için kullandığımız bir fonks.
age_cols = [col for col in df.columns if "Age" in col]


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)  # grafik daha güzel gözüksün diye yaptık


for col in age_cols:
    num_summary(df, col, plot=True)

# değişiklik var mı diye bakıyoruz
# Yapacak olduğumuz işlemlerde değişkenlere eşit yaklaşılmasını sağlamaya çalışmak demek, onların yapısını bozmak demek
# değildir. Yapılarını koruyacak şekilde ifade ediliş tarzlarını değiştirmektir.

########################################
# Numeric to Categorical
# Binning
########################################
# Sayısal değişkenleri kategorik değişkenşere çevirme

df["Age_qcut"] = pd.qcut(df["Age"], 5)
df.head()
# qcut metodu bir değişkenin değerlerini küçükten büyüğe sıralar ve çeyrek değerlere göre 5 parçaya böler

########################################################################################################################
#                                                                                                                      #
#                                 Feature Extraction (Özellik Çıkarımı)                                                #
#                                                                                                                      #
########################################################################################################################
# Ham veriden değişken üretmek.
"""
Makine öğrenmesi, derin öğrenme, zaman serisi problemleri ve bununla ilişkili birçok modelleme süreçlerinin temelini
buradaki kritik eşik noktası yönlendirmektedir, belirtmektedir. Yani benim feature türetmem lazım tabular data da olsa
resim(görüntü) verisi de olsa metinsel veri de olsa bunların üzerinden featurelar türeterek bunları temsil edeyim ki buna
göre bir sınıflandırma problemi, regresyon problemi yada zaman serisi problemi gibi problemleri çözebiliyor olayım.

Feature map: Yapısal olmayan veriyi artık bizim matematiksel işlemlere sokabileceğimiz şekilde matematiksel bir forma 
çevirmiş olma drumu söz konusu olur.

"""

#############################################
# Binary Features: Flag, Bool, True-False
#############################################
# Bu konuda genel kapsayıcı net bir literatür yoktur. Çünkü özellik çıkarımı işi problemden probleme değişebilir. Özelinde
# yöntemden yönteme değişebilir. Özetle burada yapacağımız işlem 1-0 şeklinde var olan değişkenler üzerinden yeni değişkenler türetmek.
# label encodingle karıştırmayalım. Yeni bir değişken türetmekten bahsediyoruz var olanı değiştirmek değil.

# NAN gördüğüm yere 1 olmayan yerlere 0 koymak istiyorum. notnull() -> null değil mi yani dolu mu diye soruyor
df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int")

df.groupby("NEW_CABIN_BOOL").agg(
    {"Survived": "mean"})  # cabin numarası olan ve olmayanların hayata kalma sayısına baktık.
# yorumlarsak sonucu cabin numarası olanların olmayanlara göre ciddi bir kurtulma oranı olduğunu gördük.

# oluşturduğumuz feature'ın bağımlı değişkenle olan ilişkisini merak ediyorum ve bunun için oran testi yapıyoruz.
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# t-1 ve p-2 oranları arasında bir fark yoktur der. t-1 ve p-1 neyi ifade ediyor diye soracak olursak cabin numarası olup
# hayatta kalanlar ve cabin numarası olmayıp hayatta kalanları ifade ediyor. İkisi arasında fark yoktur diyen H:0 hipotezi
# p-value değeri 0.05'ten küçük olduğundan dolayı reddedilir. Yani aralarında istatistiki olarak  anlamlı bir farklılık
# var gibi gözüküyor.

# çok değişkenli etkiyi bilmiyorum şuan sadece iki etmene göre kıyas yapıyoruz. Ama şuan da anlamlı olmasıyla ilgili
# yeterli delili buldum.


# Başka binary featurlar daha oluşturalım.
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


#############################################
# Text Features: Text'ler Üzerinden Özellik Türetmek
#############################################
df.head()

#  - Letter Count: Harfleri Saydırma
df["NEW_NAME_COUNT"] = df["Name"].str.len()

#  - Word Count: Kelimeleri Saydırma
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))  # apply ve lambda ile name'i tuttuk
# boşluklara göre split et kaç tane kelime varsa say örneğin Braund, Mr. Owen Harris ---> 4 olarak cevap vermesi

#  - Özel Yapıları Yakalamak
# İsimlerde dr gibi meslek belirtir kelimeler vardı. başında Dr ifadesi varsa bunu seç len'nini al

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("DR")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

#############################################
# Regex Features: Regex ile Değişken Türetmek
#############################################
df.head()
# extract(çıkar) diyorum önünde boşluk sonunda nokta olucak ve büyük yada küçük harflerden oluşacak şekilde göreceğin bu ifadeleri yakala diyoruz.
df["NEW_TITLE"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#############################################
# Date Features: Date değişkenleri üretmek
#############################################

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff["year"] = dff["Timestamp"].dt.year

# month
dff["month"] = dff["Timestamp"].dt.month

# year diff
dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff["month_diff"] = (date.today().year - dff["Timestamp"].dt.year) * 12 + date.today().month - dff["Timestamp"].dt.month

# day name
dff["day_name"] = dff["Timestamp"].dt.day_name()

##################################################
# Feature Interaction ( Özellik Etkileşimleri )
##################################################
# Değişkenlerin birbiriyle etkileşime girmesi yani örneğin 2 değişkenin çarpılması yada toplanması yada karesinin alınması gibi
df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]  # ikisinin de standart formda olması varsayımı altındadır.

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "male") & ((df["Age"] > 21) & (df["Age"]) <= 50), "NEW_SEX_CAT"] = "maturemale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniormale"

df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngfemale"

df.loc[(df["Sex"] == "female") & ((df["Age"] > 21) & (df["Age"]) <= 50), "NEW_SEX_CAT"] = "maturefemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()


########################################################################################################################
#                                                                                                                      #
#                                                   Uygulama                                                           #
#                                                                                                                      #
########################################################################################################################

#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################

df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)


df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))


df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#############################################
# 4. Label Encoding
#############################################

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


#############################################
# 5. Rare Encoding
#############################################

rare_analyser(df, "SURVIVED", cat_cols)


df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#############################################
# 6. One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

#############################################
# 8.Model
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)

