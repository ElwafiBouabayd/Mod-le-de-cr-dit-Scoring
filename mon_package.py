import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import category_encoders as ce 
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, precision_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder

def lire_data(path, drop_columns):
    df = pd.read_csv(path).drop(columns=drop_columns)
    return df

def comprendre_data(df):
    print(f"Colonnes numériques : {list(df.select_dtypes(include=['int64','float64']))}")
    print(f"Colonnes catégorielles : {list(df.select_dtypes(include='object'))}")
    print(f"Colonnes sans variance:{df.drop(columns = 'User_id')._get_numeric_data().loc[:, df.drop(columns = 'User_id')._get_numeric_data().std() == 0].columns}")

def statistiques_descriptives(df):
    return df.drop(columns = 'User_id')._get_numeric_data().describe(), df.drop(columns = 'User_id').select_dtypes(include= 'object').describe()

def analyse_DPD(df):
    dpd_flow = pd.DataFrame(columns=["Jours de retard","Nombre des clients"])
    for dpd in [0,30,60,90]:
        user_count = len(df[df.max_dpd>=dpd])
        dpd_flow.loc[len(dpd_flow.index)] = [dpd, user_count]
    dpd_flow['Nombre des clients']  = dpd_flow['Nombre des clients'].astype(int)
    dpd_flow['Pourcentage des clients'] = (dpd_flow['Nombre des clients']*100/max(dpd_flow['Nombre des clients'])).round(2).astype(str)+' %'
    return dpd_flow
# à quelle EMI le client dépasse le seuil dpd pour la première fois.
def analyse_EMI(df, dpd):
    df2 = df[df.max_dpd>=dpd]
    df2['premier dépassement'] = np.where(df2.emi_1_dpd>=dpd, 1, 
                                   np.where(df2.emi_2_dpd>=dpd, 2, 
                                           np.where(df2.emi_3_dpd>=dpd, 3, 
                                                   np.where(df2.emi_4_dpd>=dpd, 4, 
                                                           np.where(df2.emi_5_dpd>=dpd, 5, 
                                                                   np.where(df2.emi_6_dpd>=dpd, 6, 0))))))
    window_roll = df2.groupby('premier dépassement')['User_id'].count().reset_index()
    window_roll['user_percent'] = (window_roll['User_id']*100/sum(window_roll['User_id'])).round(2).astype(str)+' %'
    window_roll.columns = ['Première mensualité en défaut','Nombres des clients en défaut', 'Pourcentage des clients']
    return window_roll

def ajouter_label(df,dpd, mois):
    mois = ["emi_"+str(x)+"_dpd" for x in range(1, moi+1)]
    df['label'] = np.where(df[months].max(axis = 1)>=dpd, 1, 0)
    print("label columns added to dataframe")
    return df
    