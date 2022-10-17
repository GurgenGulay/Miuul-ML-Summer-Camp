# Makine Öğrenmesi Nedir?
"""
Bilgisayarların insanlara benzer şekilde öğrenmesini sağlamak maksadıyla çeşitli algoritma ve tekniklerin geliştirilmesi
için çalışan bilimsel çalışma alanıdır.

Bağımlı değişkeni yani hedeflediğimiz değişkeni sayısal olan problemlere regresyon problemleri denir.
Bağımlı değişken kategorik bir değişken olduğunda yani sınıfları ifade eden bir değişken olduğunda bu bir sınıflandırma
problemidir.
"""

########################################################################################################################
#                                                                                                                      #
#                                            Temel Kavramlar                                                           #
#                                                                                                                      #
########################################################################################################################

#########################################
# Değişken Türleri  ( Variable Types )
##########################################
"""
    - Sayısal Değişkenler
    - Kategorik Değişkenler ( Nominal, Ordinal )
    - Bağımlı Değişken ( target, dependent, output, response )
    - Bağımsız Değişken ( feature, independent, input, column, predictor, explanatory )
"""
#########################################
# Öğrenme Türleri ( Learning Types )
##########################################
"""
3'e ayrılır.
    1. Denetimli Öğrenme (Supervised Learning)
    2. Denetimsiz Öğrenme (Unsupervised Learning)
    3. Pekiştirmeli Öğrenme (Reinforcement Learning)
    
 -Denetimli Öğrenme (Supervised Learning)
Eğer veri setinde labellarımız yer alıyorsa bu durumda veri denetimli öğrenmedir. Üzerinde çalışılan verilerde eğer bir
bağımlı değişken varsa bir target varsa bu durumda bu bir denetimli öğrenme problemindir. Bağımlı bağımsız değişkenin 
arasındaki ilişki öğreniliyor olur.

 -Denetimsiz Öğrenme (Unsupervised Learning)
İlgili veri setlerinde label, target, hedef ve bağımlı değişkende olmadığı durumlar bir denetimsiz öğrenme problemidir.
"""

#########################################
# Problem Türleri ( Problem Types )
##########################################
"""
 * Regresyon problemlerinde bağımlı değişken sayısal,
 * Sınıflandırma problemlerinde bağımlı değişken kategoriktir.
"""
#########################################
# Model Başarı Değerlendirme Yöntemleri
##########################################
"""
Tahminlerim ne kadar başarılı?

Regresyon Modellerinde Başarı Değerlendirme
- Ortalama hata yani MSE formülü, Mse değeri ne kadar küçükse o kadar iyidir.
- RMSE(MSE'nin karekökü alınmış halidir.)  
- MAE(Mutlak ortalama hata)

Sınıflandırma Modellerinde Başarı Değerlendirme
- Accuracy = Doğru Sınıflandırma Sayısı / Toplam Sınıflandırılan Gözlem Sayısı (ne kadar yüksekse o kadar iyidir.)
"""
#####################################################
# Model Doğrulama Yöntemleri ( Model Validation )
#####################################################
"""
Holdout Yöntemi (Sınama Seti Yöntemi)
Amaç şudur veri seti; eğitim seti ve test seti olarak ikiye bölünür. Eğitim seti üzerinde modelleme işlemi(train işlemi,
eğitim işlemi) gerçekleşir model burada öğrenir daha sonra bu modele test setinden sorular sorulur kendini burada test
eder ve iki başarı bu şekilde değerlendirilir.

K-Katlı Çapraz Doğrulama ( K Fold Cross Validation )
Veri setimiz bol ise bu durumda veriyi baştan ikiye bölüp cross validation yöntemini eğitim seti üzerinde bütün işlemler
bittikten sonra en son test seti üzerinde modeli test etmek şeklinde kullanılabilir. Bunu yaparken eğitim setini 5'e 
bölüp her bir itarasyonunda 4'ü ile eğitim biri ile test yaparız böylece çapraz doğrulama işlemi eğitim seti üzerinde 
gerçekleştirilir. Burada gerekli hiperparametre optimizasyonu, değişken mühendislikleri, model ayarlamaları gerçekleştirilir
ve en son hiç görmedi veriyi (test setini) tekrar girilir.

- Direkt tüm veri setini 5'e bölüp çapraz doğrulamayı orda da yapabiliriz ama bu genelde az veri olduğunda yapılır ki 
az veri olduğunda holdout yöntemide kullanılabilir.
"""
##################################################################
#   Yanlılık-Varyans Değiş Tokuşu ( Bias-Variance Tradeoff )
##################################################################
"""
 - Overfitting ( Aşırı Öğrenme, Yüksek Varyans )
Modelin veriyi öğrenmesidir. Model veriyi değil veri içindeki örüntüyü, yapıyı öğrenmelidir. Overfitting, aşırı öğrenme
modelin veriyi ezberlemesidir.
 - Underfitting ( AZ Öğrenme, Yüksek yanlılık )
Modelin veriyi öğrenememesi durumudur.
 - Doğru Model ( Düşük Yanlılık - Düşük Varyans )
Modelin veri setinin oluştuğu örüntüyü öğrenmesidir.

Model Kurmak ne demektir?
    Bağımlı ve bağımsız değişkenler arasındaki ilişkiyi ilişkinin özütünü çıkarmak, bu ilişkiyi öğrenmek demektir.

Aşırı öğrenme nasıl tespit edilir?
    Eğitim seti ve test setindeki hata değişimleri incelenir, bu iki hatanın birbirinden ayrılmaya başladığı(çatalanmaya başladığı)
nokta(Optimum Nokta) itibariyle aşırı öğrenme başlamıştır denir. 

Aşırı öğrenme probleminin nasıl önüne geçebiliriz ?   
    Aşırı öğrenmenin başladığı noktayı yakalayabiliyorsak; eğitim süresini, model karmaşıklığını, iterasyon sayısnı
durdurursak bu durumda problemin önüne geçeriz.

Model Karmaşıklığı nedir?
Model karmaşıklığını bir miktara kadar artması iyidir.
Model karmaşıklığını algoritmaların çalışma prensibleri üzerinden örnekler ile ele alalım.
    * Random Forest algoritmasında dallanma sayısının artması model karmaşıklığını ifade etmektedir.
    * LightGBM algoritmasında model karmaşıklığı iterasyon sayısına karşılık gelmektedir.
    * Doğrusal regresyon problemlerinde üstel terimlerin artması model karmaşıklığına sebep olabilmektedir.    
    
"""
########################################################################################################################
#                                                                                                                      #
#                                  Doğrusal Regresyon ( Linear Regression )                                            #
#                                                                                                                      #
########################################################################################################################
"""
Amaç, bağımlı ve bağımsız değişken/değişkenler arasındaki ilişkiyi doğrusal olarak modellemektir.
Yi = b + w*Xi
"""
###############################"
# Ağırlıkların Bulunması
##############################
"""
Gerçek değerler ile tahmin edilen değerler arasındaki farkların karelerinin toplamı/ortalamasını minumum yapabilecek
b ve w değerlerini bularak.

Model dediğimiz aslında grafikteki kırmızı doğrudur ve doğrunun formülü de Yi = b + w*Xi 'dir. 
Tahmin fonksiyonunun yükseklik aldığı değer beta, eğimini aldığı değer ise w'dur. Dolayısıyla doğrunun nereye koyulacağı 
b ve w değerlerine bağlıdır.

Ortalama hata(MSE) formülü  cost(b,w) ve örnek çözümleri defterime yazdım.

Bir modelleme çalışmasında bunlardan hangisini kullanmalıyım ?
    Yanlış bir sorudur. Örneğin MAE daha düşük çıktı o zaman bunu kullanmalıyım demek hatalıdır. Çünkü bu bir tercih 
meselesidir ve biz modelin başarısını yaptığı hatalara göre değerlendirmeye çalışıyoruz. Bu yüzden birini seçip ona göre
model üzerinde fikir üretilmelidir. Zaten bizim amacımız hatayı düşürmektir.

RMSE, MSE'nin karekök içine alınmış halidir. MAE da ise mutlaklarını alırız. Formülleri defterde.

"""

#############################################################
# Basit Doğrusal Regresyon Modeli ( Linear Regression )
############################################################
""" Sales Prediction with Linear Regression (Satış tahmin problemini doğrusal regresyon ile inceleyeceğiz) 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: "%.2f" % x) # virgülden sonra 2 basamak göster ayarı

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################

df = pd.read_csv("machine learning datasets/advertising.csv")
df.shape

X = df[["TV"]]    # ilk aşamada bu ikisi arasındaki ilişkiyi modelliycez ve sonra grafik yardımı ile bu modellemeyi inceleyeceğiz
y = df[["sales"]]


##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)  # linearregression kütüphanesi ile modeli kolayca kurduk. Teorikte zor ama kod kısmı kolay.

# y_hat (yi şapkalı hali) = b + w*x # formülümüzü hatırlayalım

# sabit (b - bias)
reg_model.intercept_[0]  # b sabitimizi getirelim 0 dememizin sebebi sadece intercept çağırdığımızda array olarak getiriyor sayıyı vermesi için indeks bilgisini giriyoruz.

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]


##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir? Çözüm (yi^ =  b + w*x)

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik tv harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T  # gördüğümüz gibi veri setinde olmayan bir değer bile olsa artık öğrendiği için bunun ne kadar satabileceğini tahmin edebilirim

#################################
# Modelin Görselleştirilmesi
##################################

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},  # kullandığımız metod regplot. x bağımsız, y bağımlı değişken ve grafiğin renkleri girilmiş
                ci=False, color="r")  # güven aralığı argümanı false yani ekleme dedik, regresyon çizgisinin renginin ne olacağı verilmiş

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")  # ,2 ifadesi ,'den sonra 2 basamak al demektir
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)   # x ekseninini -10, dan 310 kadar yap
plt.ylim(bottom=0)  # y ekseninin 0 dan başlat
plt.show()


##########################
# Tahmin Başarısı
##########################

# MSE
y_pred = reg_model.predict(X)  # Bağımsız değişkenleri modele sordum ve bağımlı değişkeni tahmin etmesini bekliyoruz. tahmin edilen değerlere y_pred dedik
mean_squared_error(y, y_pred) # mean_squared_error metoduna tahmin edilen değerleri gönderirsem ortalama hatayı verecektir.
# 10.51

y.mean()  # satışların ortalamasına ve standart sapmasına bakalım. Bakma sebebimiz ortalama hatanın ne kadar büyük olduğunu anlamaya çalışmak için
y.std() # Çıkaracağımız sonuç satışların ort. 14 birim ben bir tahmin yaptığımda 10 birim ile hata yapıyorsam o kadar da küçük bir hata değil yüksek bir hata oranımız var

# RMSE
np.sqrt(mean_squared_error(y, y_pred))  # RMSE hesaplayalım
# 3.24

# MAE
mean_absolute_error(y, y_pred)  # MAE heaplayalım
# 2.54

# R-KARE
""" 
Değişken sayısı arttıkça r-kare şişmeye mehillidir. Burada düzeltilmiş r-kare değerininde göz önünde bulundurluması gerekir.
İstatistiki çıktılarla ilgilenmiyoruz. Model anlamlılıklar, kat sayı testleri vb. yapmıyoruz. Makine öğrenmesi açısından 
ele alıyoruz. Dolayısıyla bizim için doğrusal bir formda tahmin etme görevi var ve bu tahmini en yüksek başarı ile elde 
etmeye çalışacağız
"""

reg_model.score(X, y)  # Veri setindeki bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir. Yani bu modelde bağımsızlar bağımlı değişkenin %61'ni açıklayabilmektedir.
# 0.61

#####################################################################
# Multiple Linear Regression (  Çoklu Doğrusal Regresyon Modeli )
#####################################################################

df = pd.read_csv("machine learning datasets/advertising.csv")

X = df.drop('sales', axis=1)  # bağımlı değişkenleri seçelim

y = df[["sales"]]     # bağımsız değişkenleri seçelim


##########################
# Model
##########################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
"""
train_test_split metodunu getirerek diyoruz ki bağımsız değişkenleri ve ayırdığımız bağımlı değişkeni al test setinin boyutunu 
%20 (test_size=0.20) yap ve train setinin boyutunu %80 yaparak rastgele bir örneklem oluştur(train ve test setlerini oluştur) diyoruz.
random_state=1 şunu ifade ediyor dersi anlatan hocanın örneklemi ile aynı olmasını istiyorsak 1 yazıyoruz. Aynı sayı 
yazıldıktan sonra aynı rassallıkta train, testler oluşacağından dolayı aynı sonuçları almayı sürdürüyor olacağız. Burada 7 de
yazabilirdi o zaman hoca da 10 yazsaydı aynısı olurdu.
"""
# train setiyle model kurucaz, test setiyle kontrol edicez

y_test.shape
y_train.shape

reg_model = LinearRegression().fit(X_train, y_train)  # LinearRegression() metodu ve fit ile modeli kuralım



##########################
# Çoklu Doğrusal Regresyonda Tahmin İşlemleri
##########################

# sabit (b - bias)
reg_model.intercept_
# 2.90

# coefficients (w - weights)
reg_model.coef_
# 0.0468431 , 0.17854434, 0.00258619

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40


# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

2.90794702 + 30 * 0.0468431 + 10 * 0.17854434 + 40 * 0.00258619

# Yukarıda yaptığımızı fonksiyonlaştıralım

yeni_veri = [[30], [10], [40]]  # 30, 10 ve 40 birimi ifade eden liste oluşturdum
yeni_veri = pd.DataFrame(yeni_veri).T  # sonra bunu dataframe çeviriyorum

reg_model.predict(yeni_veri)  # reg_model de  predict yani tahmin et diyorum bağımsız değişkeni(yeni_veri) veriyoruz ve bize 6.202131 cevabını veriyor.

##########################
# Tahmin Başarısını Değerlendirme
##########################
# Buradan anladığımız şey yeni değişken eklendiğinde başarının arttığı hatanın düştüğüdür.

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# TRAIN RKARE
reg_model.score(X_train, y_train)

# Test RMSE
y_pred = reg_model.predict(X_test)   # train üzerinden kurduğumuz modele test setinden soruyoruz. predict'e test setinin x'leri yani test setinin bağımsız değişkenlerini soruyoruz modelle
# o da test setinin bağımlı değişkenini tahmin ediyor.
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41  Normalde test hatası, train hatasından daha yüksek çıkar. Dolayısıyla önümüzde ki tablo iyi beklenti dışı güzel bir durum.

# Test RKARE
reg_model.score(X_test, y_test)


# 10 Katlı CV RMSE   (çapraz doğrulama ile)
np.mean(np.sqrt(-cross_val_score(reg_model,   # cross_val_score "-" değerler getirdiği için - ile çarpıyoruz
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1.69


# 5 Katlı CV RMSE  (çapraz doğrulama ile)
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71




######################################################
# Simple Linear Regression with Gradient Descent from Scratch ( Gradient Descent ile Doğrusal Regresyon )
######################################################
# Bonus bölüm

# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)  # m = gözlem sayısı
    sse = 0  # sse = hata kareler toplamı

    for i in range(0, m):
        y_hat = b + w * X[i]  # y_hat = tahmin edilen değişkenler
        y = Y[i]   # y = gerçek değerler
        sse += (y_hat - y) ** 2 # karelerini topla dedik

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0  # w türevinin toplamı
    w_deriv_sum = 0   # b türevinin toplamı
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)  # teta 0'ın kısmi türevi
        w_deriv_sum += (y_hat - y) * X[i]  # teta 1'in kısmi türevi
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("machine learning datasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters (veri setinden bulunamayan ve kullanıcı tarafından ayarlanması gereken parametrelerdir )
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)

# Çıktı: After 100000 iterations b = 9.311638095155203, w = 0.2024957833925339, mse = 18.09239774512544



























































































































































































































































































































