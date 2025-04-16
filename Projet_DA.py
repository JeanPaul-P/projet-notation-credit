#%%
#Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
#Charger les données 
df = pd.read_csv("transactions_bancaires.csv", parse_dates= ["date"])

#Aperçu des données 
df.head()
df.info()

# %%
# Vérifier les valeurs manquantes
total_missing = df.isnull().sum()
print("Valeurs manquantes:\n", total_missing)

#%%
#Vérifier les doublons 
duplicates = df.duplicated().sum()
print("Nombre de doublons:", duplicates)

# %%
#Analyse statistique (moyenne, médiane, distribution des montants)
descriptive_stats = df["montant"].describe()
print("Statistiques descriptives:\n", descriptive_stats)

# %%
#Histogramme distribution des montants
plt.figure(figsize=(10, 5))
plt.hist(df["montant"], bins=30, color='royalblue', alpha=0.6)
# Ajouter des titres et labels
plt.title("Distribution des montants", fontsize=14)
plt.xlabel("Valeurs", fontsize=12)
plt.ylabel("Fréquence", fontsize=12)
# Afficher le graphique
plt.show()

# %%
# Liste des valeurs uniques dans 'type_transaction'
transaction_types = df["type_transaction"].unique()
print("Types de transaction:\n", transaction_types)

# Sauvegarde des types de transaction dans un fichier Excel
transaction_types_df = pd.DataFrame(transaction_types, columns=["type_transaction"])
transaction_types_df.to_excel("types_transaction.xlsx", index=False)


# %%
#Détection des anomalies (méthode IQR)
Q1 = df["montant"].quantile(0.25) #1er quartile
Q3 = df["montant"].quantile(0.75) #2ème quartile
IQR = Q3 - Q1

val_aberrantes = df[(df["montant"] < (Q1 - 1.5 * IQR)) | (df["montant"] > (Q3 + 1.5 * IQR))]
print("Nombre d'anomalies détectées:", val_aberrantes.shape[0])

# %%
#Visualisation avec matplotlib
#Histogramme des montants de transactions
plt.figure(figsize=(10, 5))
plt.hist(df["montant"], bins=50, color='blue', alpha=0.7)
plt.xlabel("Montant de la transaction")
plt.ylabel("Fréquence")
plt.title("Distribution des montants des transactions")
plt.show()

# %%
#Boxplot pour détecter les anomalies
plt.figure(figsize=(8, 4))
plt.boxplot(df["montant"], vert=False)
plt.title("Boxplot des montants de transactions")
plt.figtext(0.5, -0.1, 
            "Les points hors des moustaches du boxplot correspondent aux valeurs extrêmes détectées par la méthode IQR \n"
            "Cette méthode permet d'identifier des transactions suspectes qui pourraient indiquer des erreurs ou des fraudes (ex : retraits ou virements anormalement élevés)", 
            wrap=True, horizontalalignment="center", fontsize=10)
plt.tight_layout()  # Ajuster l'espacement

plt.show()

# %%
#Courbe d’évolution du solde moyen des clients
    #Cumule des montants par client
df["solde_cumul"] = df.groupby("client_id")["montant"].cumsum()
    #Moyenne quotidienne de ce cumule
df_avg_solde = df.groupby("date")["solde_cumul"].mean()
print(df_avg_solde)

plt.figure(figsize=(12, 6))
plt.plot(df_avg_solde.index, df_avg_solde.values, marker='o', linestyle='-', color='b')
plt.xlabel("Date")
plt.ylabel("Solde moyen")
plt.title("Évolution du solde moyen des clients")
plt.grid()
plt.show()


# %%
#Courbe d’évolution du solde moyen des clients PAR DATE YYYY-MM-DD
    #Date format YYYY-MM-DD
df['date_jour'] = df['date'].dt.date 
print(df['date_jour'])
    #Aggrège par date 
df_grouped = df.groupby('date_jour').size().reset_index(name='count')
print(df_grouped)

# %%
    #Cumule des montants par client
df["solde_cumul2"] = df.groupby("client_id")["montant"].cumsum()
    #Moyenne quotidienne de ce cumule
df_avg_solde2 = df.groupby("date_jour")["solde_cumul2"].mean()
print(df_avg_solde2)

    #Graph
plt.figure(figsize=(12, 6))
plt.plot(df_avg_solde2.index, df_avg_solde2.values, marker='o', linestyle='-', color='b')
plt.xlabel("Date")
plt.ylabel("Solde moyen")
plt.title("Évolution du solde moyen des clients")
plt.grid()
plt.show()

# %%
df.drop(columns=['avg_solde'], inplace=True)

# %%
df.head()
# %%
# Sauvegarde des données nettoyées pour Power BI
df.to_csv("transactions_nettoyees.csv", index=False)

# %%
df.info()

# %%
df['montant'] = df['montant'].astype('int64')
df.head()
# %%
df['solde_cumul'] = df['solde_cumul'].astype('int64')
df['solde_cumul2'] = df['solde_cumul2'].astype('int64')
df['solde_apres_transaction'] = df['solde_apres_transaction'].astype('int64')

df.head()
# %%
# Sauvegarde des données nettoyées pour Power BI
df.to_csv("transactions_nettoyees.csv", index=False)

# %%
