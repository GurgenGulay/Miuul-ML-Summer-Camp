#####################################
# Fonksiyonlara Özellik Eklemek
#####################################
"""
Görev: cat_summary() fonksiyonuna 1 özellik ekleyiniz. Bu özellik argümanla biçimlendirilebilir olsun. Var olan
özelliği de argümanla kontrol edilebilir hale getirebilirsiniz.
"""


def check_df(dataframe, head=5, extra_test=False):
    print("################## Shape ##################") 
    print(dataframe.shape)
    print("################## Types ##################")
    print(dataframe.dtypes)
    print("################## Head ##################")
    print(dataframe.head(head))
    print("################## Tail ##################")
    print(dataframe.tail(head))
    if extra_test:
        print("################## NA ##################")
        print(dataframe.isnull().sum())
        print("################## Quantiles ##################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df, 5, True)

#######################
# Docstring Yazımı
#######################
"""
Görev: check_df(), cat_summary() fonksiyonlarına 4 bilgi (uygunsa) barındıran numpy tarzı docstring 
yazınız. (task, params, return, example)
"""

def check_df(dataframe, head=5):
    """
    Checks the dataframe.
    Parameters:
    ------
    dataframe : DataFrame
        Kullanıcının tercih ettiği herhangi bir veri seti olabilir.
    head : int > 0
        Veri setinin başından başlayarak çıktıda görülmesi istenen satır sayısı.
    Returns:
    ------
    None
    """
    print("################## Shape ##################")
    print(dataframe.shape)
    print("################## Types ##################")
    print(dataframe.dtypes)
    print("################## Head ##################")
    print(dataframe.head(head))
    print("################## Tail ##################")
    print(dataframe.tail(head))
    print("################## NA ##################")
    print(dataframe.isnull().sum())
    print("################## Quantiles ##################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# check_df(df, head=5)

def cat_summary(dataframe, col_name, plot=False):
    """
    Parameters
    ----------
    dataframe : DataFrame
        Kullanıcının tercih ettiği herhangi bir veri seti olabilir.
    col_name : str
        Veri setinin sütunlarından biri.
    plot : bool
        Veri setinin bir grafiğini almak istiyorsanız True olarak ayarlayın.
    Returns
    -------
    None
    """

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# cat_summary(df, "total", plot=True)








