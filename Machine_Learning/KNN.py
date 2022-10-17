################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("Machine_Learning/machine learning datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################
"""
KNN yöntemi uzaklık temelli bir yöntemdir. Uzaklık temelli yöntemlerde ve Gradient Descent temelli yöntemlerde değişkenlerin
standart olması elde edilecek sonuçların ya daha hızlı yada daha doğru olmasını sağlayacaktır. Özetle daha başarılı olmasını 
sağlayacaktır. Bu yüzden bu değişkenleri standartlaştırma işlemine sokuyor olacağız.
"""
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X)  # numpy arrayi döner ve elimizde sütun isimleri yok

X = pd.DataFrame(X_scaled, columns=X.columns)  # ölçeklendirilmiş x'leri alıp bunu bir df çeviriyoruz ve sütunlarını bu
# bağımsız değişkenin ilk halinden alarak (X.columns) burada sütun ismi olarak giriyoruz.

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)  # bağımsız ve bağımlı değişkenleri giriyoruz

# tahmin yapalım
random_user = X.sample(1, random_state=45) # (1, ) -> 1 tane örneklem seç

knn_model.predict(random_user)  # predict metodu aracılık yapar ve bizim içine yazıp öğrenmek istediğimizi modele gidip sorar.

# Tek bir gözlem birimi için tahmin de bulunduk, bütün gözlem birimi için tahminde bulunup başarımızın ne durumda old. değerlendirelim.
################################################
# 4. Model Evaluation
################################################

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1] # 0. indeksi değil 1.indeksi istiyoruz çünkü 1.sınıfa ait olma olasılıkları ile ilgileniyoruz.

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.74
# AUC
roc_auc_score(y, y_prob)
# 0.90

# 5 katlı çapraz doğrulama yapalım.
cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])  # scoring kullanmak istediğimiz metrikler

# score_time tahmin süresi
#  'test_roc_auc': array([0.77555556, 0.78759259, 0.73194444, 0.83226415, 0.77528302])}
# 5 katlı çapraz doğrulama yaptığımızda veri setini 5'e böldü, 4 tanesi ile model kurdu 1 tanesi ile test etti. İlk yaptığı
# işlemde elinde 1 tane hata var 0.77555556 Sonra tekrar 4 tanesi ile model kurup 1'yle test etti ve 2.hatayı buldu 78759259 .....

# 5 katlı çapraz doğrulamanın bütün test scorelarının ortalamasını alıcaz.
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# 0.73
# 0.59
# 0.78

# Başarı score'ları nasıl arttırılabilir?
# 1. Örnek (veri) boyutu arttıralabilir.
# 2. Veri ön işleme işlemleri detaylandırılabilir.
# 3. Özellik mühendisliği - yeni değişkenler türetilebilir.
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()
"""
Parametre ve hiperparatme arasındaki fark ?
Parametre, modellerin veri içinden öğrendiği ağırlıklardır. Ağırlıklar parametrelerin tahmincileridir.

Hiperparametre, kullanıcı tarafından tanımlanması gereken dışsal ve veri seti içerisinden öğrenilemeyen parametrelerdir.
"""
################################################
# 5. Hyperparameter Optimization
################################################

# Şuan komşuluk sayımız n_neighbors': 5, amacımız bu komşuluk sayısını değiştirerek olması gereken en optimum komşuluk
# sayısının ne olacağını bulmak.
knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)} # sözlük içerisinde bir parametre var ve 2'den 50'ye kadar sayılar oluşturdum
# her seferinde kat sayısını değistirip cv uygulayacak. Fonksiyonda cv=5 yazmamızın sebebi 2'den 50'ye dediğimizde sırasıyla
# gelen örneklemlerde 5 katlı çapraz doğrulama yapacak.
knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,   # -1 yapıldığında işlemcileri tam performans kullanır.
                           verbose=1).fit(X, y)   # verbose rapor içindir 1 yaptığımızda rapor çıkarır

# Kodu çalıştırdığımızda aşağıdaki gibi bilgi mesajı geliyor 48 tane denenecek hiperparametre değeri var ve bu herbir
# parametre için 5 katlı cv yapılacağından dolayı toplam 240 tane model kurma işlemi vardır.
# Fitting 5 folds for each of 48 candidates, totalling 240 fits

knn_gs_best.best_params_
# {'n_neighbors': 17}

################################################
# 6. Final Model
################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)  # bulduğumuz en iyi komşuluk değerini set ediyoruz. ** ile otomatik atama yapılır.

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)
