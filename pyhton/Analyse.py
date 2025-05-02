#%% 
#Synchronisation du fichier .ipnyb avec le fichier .py
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
# ---

#%%
#Importation des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
#CHARGEMENT DES DONNEES

# %%
#Chargement des datas
df = pd.read_excel(r"C:\Users\Jppok\venv\test_pandas\Projet DA_Portfolio\data\CorporateCreditRating.xlsx")

#Afficher les 1ères lignes
print(df.head())

#Vérifier la structure des données
print(df.info())

#%%
#Chargement de la table détail des secteurs 
df_Sector = pd.read_excel(r"C:\Users\Jppok\venv\test_pandas\Projet DA_Portfolio\data\Sector_Table.xlsx")

#Afficher les 1ères lignes
print(df_Sector.head())

#Vérifier la structure des données
print(df_Sector.info())


#%%
#PREPARATION DES DONNEES
#%%
#Valeurs manquantes et doublons
    #valeurs manquantes
print(df_Sector.isnull().sum())
    #doublons
doublons = df_Sector.duplicated().sum()
print("Nombre de doublons:", doublons)

#%%
#Ajout de la table détail des secteurs à df
df = pd.merge(df, df_Sector, on="SIC Code", how="left")

df.head()
#%%
#Valeurs manquantes
print(df.isnull().sum())
#%%
#Suppression des lignes où SubSector est vide
df = df.dropna(subset=["SubSector"])
print(df.isnull().sum())

#%%
#doublons
doublons = df.duplicated().sum()
print("Nombre de doublons:", doublons)

#%% 
#Renommage de "Binary Rating" = Investment Grade 
df.rename(columns={'Binary Rating': 'Investment Grade'}, inplace=True)

df.columns
#%%
#Type de données 
print(df.dtypes)

#%%
#Liste des secteurs uniques
df["Sector"].unique()

#%%
#Renommer les secteurs
sector_names = {
    "BusEq": "Business Equipment",
    "Chems": "Chemicals",
    "Durbl": "Durables",
    "Enrgy": "Energy",
    "Hlth": "Health",
    "Manuf": "Manufacturing",
    "Money": "Money",
    "NoDur": "Non-Durables",
    "Other": "Other",
    "Shops": "Shops",
    "Telcm": "Telecommunications",
    "Utils": "Utilities"}

df["Sector"] = df["Sector"].replace(sector_names)

#Vérification des changements
df["Sector"].unique()

#%%
#ANALYSE EXPLORATOIRE DES DONNEES
# %%
#Affichage de la répartition des notations de crédit
df["Rating"].value_counts().plot(kind="bar", figsize=(8, 4))
    #Ajouter des titres et labels
plt.title("Répartition des notations de crédit")
plt.xlabel("Notation")
plt.ylabel("Nombre d'entreprises")
    #Afficher le graphique
plt.show()

#Distribution des notations de crédit
df["Rating"].hist(figsize=(12, 8), bins=30)

plt.title("Distribution des notations de crédit")
plt.xlabel("Notation")
plt.ylabel("Nombre d'entreprises")
plt.show()

#%%
#Analyse des ratios financiers 
#%%
#Sélection des ratios 
ratios = ["Current Ratio", "Debt/Equity Ratio", "ROE - Return On Equity"]

#histogrammes
plt.figure(figsize=(15, 5))
for i, ratio in enumerate(ratios, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[ratio], bins=30, kde=True)
    plt.title(f"Distribution de {ratio}")
    plt.xlabel(ratio)
    plt.ylabel("Fréquence")
plt.tight_layout()
plt.show()

#Statistique descriptive
df[ratios].describe()

#%%
#Analyse écart interquartile
total_rows = df.shape[0]  

for i, ratio in enumerate(ratios, 1):
    Q1 = df[ratio].quantile(0.25) #1er quartile
    Q3 = df[ratio].quantile(0.75) #2ème quartile
    IQR = Q3 - Q1

    #Filtrage des valeurs aberrantes
    val_aberrantes = df[(df[ratio] < (Q1 - 1.5 * IQR)) | (df[ratio] > (Q3 + 1.5 * IQR))]
    
    print(f"Ratio : {ratio}")
    print(f" - Valeurs aberrantes : {val_aberrantes.shape[0]}")
    print(f" - Pourcentage du total : {round((val_aberrantes.shape[0]/total_rows)*100, 2)} %\n")

# %%
#Dispersion avec boxplot
plt.figure(figsize=(8, 6))
    #Afficher plusieurs graphs (subplot)
plt.subplot(3, 1, 1)
plt.boxplot(df["Current Ratio"], vert=False)
plt.title("Boxplot de Current Ratio")
plt.subplot(3, 1, 2)
plt.boxplot(df["Debt/Equity Ratio"], vert=False)
plt.title("Boxplot de Debt/Equity Ratio")
plt.subplot(3, 1, 3)
plt.boxplot(df["Net Profit Margin"], vert=False)
plt.title("Boxplot de Net Profit Margin")
plt.figtext(0.5, -0.1, 
            "Les points hors des moustaches du boxplot correspondent aux valeurs extrêmes détectées par la méthode IQR. \n"
            "Cette méthode permet d'identifier les potentielles valeurs aberrantes", 
            wrap=True, horizontalalignment="center", fontsize=10)
plt.tight_layout() 


# %%
#Etude des relations entre variables (corrélations, tendances).
#Convertir la notation de crédit en score numérique 
rating_mapping = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA-": 4,
    "A+": 5, "A": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB-": 10,
    "BB+": 11, "BB": 12, "BB-": 13,
    "B+": 14, "B": 15, "B-": 16,
    "CCC+": 17, "CCC": 18, "CCC-": 19,
    "CC+": 20, "CC": 21,
    "C": 22, "D": 23}
#Associe la notation au nouveau score numérique 
df["Rating_Score"] = df["Rating"].map(rating_mapping)

#%%
#Analyse de la tendance globale et par secteur du Rating score
#Tendance globale
trend = df.groupby(df["Rating Date"].dt.to_period("Y"))["Rating_Score"].mean()

#Tendance par secteur
sector_trends = df.groupby([df["Rating Date"].dt.to_period("Y"), "Sector"])["Rating_Score"].mean().unstack()

#Visualisation
sector_trends.plot(figsize=(12, 8))
trend.plot(kind="line", figsize=(12, 8), marker="o",color="black", linewidth=3, label="Tendance globale")
plt.xlabel("Année")
plt.ylabel("Score moyen de notation")
plt.title("Tendances des scores de notation par secteur et tendance globale")
plt.legend(title="Légende", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.show()

#%%
#Analyse avec la matrice de corrélation
df[["Rating_Score", "Current Ratio", "Debt/Equity Ratio", "Long-term Debt / Capital", 
          "ROE - Return On Equity", "Net Profit Margin", "EBITDA Margin", "ROI - Return On Investment", 
          "Return On Tangible Equity", "ROA - Return On Assets"]].corr()


#%%
#Filtrage de df 
df = df[["Rating Agency", "Corporation", "Rating", "Rating Date", "Investment Grade", 
         "CIK", "SIC Code", "Sector", "SubSector", "Industry", "Ticker",
         "Current Ratio", "Long-term Debt / Capital", "Debt/Equity Ratio",
         "Net Profit Margin", "ROE - Return On Equity",
         "Rating_Score"]]

df.columns

#%%
#VISUALISATION 
#%%
#Ratio de liquidité
#Nuage de point  avec une courbe de tendance
plt.figure(figsize=(10, 6))
sns.regplot(x=df["Current Ratio"], y=df["Rating_Score"], 
            scatter_kws={'alpha': 0.5}, line_kws={"color": "red"})

plt.title("Relation entre le Current Ratio et la Notation de Crédit")
plt.xlabel("Current Ratio")
plt.ylabel("Notation de Crédit (Score numérique)")
plt.grid(True)
plt.figtext(0.5, -0.1, 
            "Pour l'axe de la Notation de crédit : \n"
            "Plus le score est élevé, "
            "plus la notation est mauvaise (ex: AAA=1, CCC=18 ou plus) \n", 
            wrap=True, horizontalalignment="center", fontsize=10)

plt.show()


#%%
#Nombre d'entreprises par Rating_Score 
rating_counts = df["Rating_Score"].value_counts()

#Rating_count pour associer le nombre d'entreprise à la note de Rating score
df["Rating_Count"] = df["Rating_Score"].map(rating_counts)

# Définir la taille des points en fonction du nombre d’entreprises ayant la même note
#Racine carrée (sqrt) pour atténuer l'effet de la distribution hétérogène de Rating_score 
point_sizes = np.sqrt(df["Rating_Count"]) * 20  

#Nuage de point 
plt.figure(figsize=(10, 6))
plt.scatter(x=df["Current Ratio"], y=df["Rating_Score"], 
            s=point_sizes, c=df["Rating_Score"], alpha=0.7, cmap="coolwarm")

plt.title("Relation entre Current Ratio et la Notation de Crédit")
plt.xlabel("Current Ratio")
plt.ylabel("Notation de Crédit (Score numérique)")
plt.colorbar(label="Score de Notation (Format numérique)")
plt.grid(True)
plt.figtext(0.5, -0.05, 
            "Pour l'axe de la Notation de crédit : \n"
            "Plus le score est élevé, "
            "plus la notation est mauvaise (ex: AAA=1, CCC=18 ou plus) \n"
            "Plus une notation est fréquente, plus le point sera grand.", 
            wrap=True, horizontalalignment="center", fontsize=10)

plt.show()

#%%
#Ratio d'endettement 
#Debt/Equity Ratio
plt.figure(figsize=(10, 6))
sns.regplot(x=df["Debt/Equity Ratio"], y=df["Rating_Score"], 
            scatter_kws={'alpha': 0.5}, line_kws={"color": "red"})

plt.title("Relation entre Debt/Equity Ratio et la Notation de Crédit")
plt.xlabel("Debt/Equity Ratio")
plt.ylabel("Notation de Crédit (Score numérique)")
plt.grid(True)
plt.figtext(0.5, -0.1, 
            "Pour l'axe de la Notation de crédit : \n"
            "Plus le score est élevé, "
            "plus la notation est mauvaise (ex: AAA=1, CCC=18 ou plus) \n", 
            wrap=True, horizontalalignment="center", fontsize=10)

plt.show()

#Long-term Debt / Capital
plt.figure(figsize=(10, 6))
plt.scatter(x=df["Long-term Debt / Capital"], y=df["Rating_Score"], 
            s=point_sizes, c=df["Rating_Score"], alpha=0.7, cmap="coolwarm")

plt.title("Relation entre Long-term Debt / Capital et la Notation de Crédit")
plt.xlabel("Long-term Debt / Capital")
plt.ylabel("Notation de Crédit (Score numérique)")
plt.colorbar(label="Score de Notation (Format numérique)")
plt.grid(True)
plt.figtext(0.5, -0.05, 
            "Pour l'axe de la Notation de crédit : \n"
            "Plus le score est élevé, "
            "plus la notation est mauvaise (ex: AAA=1, CCC=18 ou plus) \n"
            "Plus une notation est fréquente, plus le point sera grand.", 
            wrap=True, horizontalalignment="center", fontsize=10)
plt.show()
#%%
#Ratio de rentabilité 
#Net Profit Margin
plt.figure(figsize=(10, 6))
sns.regplot(x=df["Net Profit Margin"], y=df["Rating_Score"], 
            scatter_kws={'alpha': 0.5}, line_kws={"color": "red"})

plt.title("Relation entre Net Profit Margin et la Notation de Crédit")
plt.xlabel("Net Profit Margin")
plt.ylabel("Notation de Crédit (Score numérique)")
plt.grid(True)
plt.figtext(0.5, -0.1, 
            "Pour l'axe de la Notation de crédit : \n"
            "Plus le score est élevé, "
            "plus la notation est mauvaise (ex: AAA=1, CCC=18 ou plus) \n", 
            wrap=True, horizontalalignment="center", fontsize=10)

plt.show()

#ROE - Return On Equity
plt.figure(figsize=(10, 6))
plt.scatter(x=df["ROE - Return On Equity"], y=df["Rating_Score"], 
            s=point_sizes, c=df["Rating_Score"], alpha=0.7, cmap="coolwarm")

plt.title("Relation entre ROE - Return On Equity et la Notation de Crédit")
plt.xlabel("ROE - Return On Equity")
plt.ylabel("Notation de Crédit (Score numérique)")
plt.colorbar(label="Score de Notation (Format numérique)")
plt.grid(True)
plt.figtext(0.5, -0.05, 
            "Pour l'axe de la Notation de crédit : \n"
            "Plus le score est élevé, "
            "plus la notation est mauvaise (ex: AAA=1, CCC=18 ou plus) \n"
            "Plus une notation est fréquente, plus le point sera grand.", 
            wrap=True, horizontalalignment="center", fontsize=10)

plt.show()

#%%
#SECTEUR
#%%
#Regroupement rating par secteur
sector_rating = df.groupby(['Sector', 'Rating']).size().unstack().fillna(0)

#Visualisation
sector_rating.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Répartition des notations de crédit par secteur")
plt.ylabel("Nombre d'entreprises")
plt.xlabel("Secteur")
plt.legend(title='Notation')
plt.tight_layout()
plt.figtext(0.5, -0.05, 
            "Les secteurs sont regroupés par notation de crédit.",
            wrap=True, horizontalalignment="center", fontsize=10)
plt.show()


#%% 
#"Binary Rating" = Investment Grade ou Non-Investment Grade 
df.rename(columns={'Binary Rating': 'Investment Grade'}, inplace=True)

df.columns
#%%
#Répartition des entreprise IG par secteur
sector_IG = df.groupby(['Sector', 'Investment Grade']).size().unstack().fillna(0)

# Visualisation
sector_IG.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title("Répartition des Investment Grade par secteur")
plt.ylabel("Nombre d'entreprises")
plt.xlabel("Secteur")
plt.legend(title='IG')
plt.tight_layout()
plt.figtext(0.5, -0.05, 
            "Les secteurs sont regroupés par Investment Grade. \n"
            "1 : Investment Grade et 0 : Non Investment Grade",
            wrap=True, horizontalalignment="center", fontsize=10)
plt.show()


#%%
#Matrice de corrélation par secteur
#Liste des secteurs uniques
sectors = df["Sector"].dropna().unique()
print(sectors)
#Calculer les corrélations par secteur
for sector in sectors:
    print(f"Corrélations pour le secteur : {sector}")
    sector_data = df[df["Sector"] == sector]
    correlation_matrix = sector_data[[
        "Current Ratio", "Debt/Equity Ratio", 
        "Net Profit Margin", "ROE - Return On Equity"]].corr()
    print(correlation_matrix)
    print("\n")

#%%
#Nuage de points pour visualiser les relations entre les ratios financiers et la notation de crédit
# Visualiser les relations par secteur
for sector in sectors:
    print(f"Visualisation pour le secteur : {sector}")
    sector_data = df[df["Sector"] == sector]
    
    # Exemple : Relation entre Current Ratio et Rating_Score
    plt.figure(figsize=(10, 6))
    sns.regplot(x=sector_data["Current Ratio"], y=sector_data["Rating_Score"], 
                scatter_kws={'alpha': 0.5}, line_kws={"color": "red"})
    plt.title(f"Relation entre Current Ratio et Notation de Crédit ({sector})")
    plt.xlabel("Current Ratio")
    plt.ylabel("Notation de Crédit (Score numérique)")
    plt.grid(True)
    plt.show()

#%%
# Moyenne des ratios financiers par secteur
sector_ratios = df.groupby('Sector')[["Current Ratio", "Debt/Equity Ratio", "Net Profit Margin", 
        "ROE - Return On Equity"]].mean().sort_values(by='Current Ratio', ascending=True)

# Visualisation avec heatmap
sns.heatmap(sector_ratios, annot=True, cmap='coolwarm')
plt.title("Moyennes des ratios financiers par secteur :")
plt.show()

#%%
sector_means = df.groupby("Sector")[[
    "Current Ratio", "Debt/Equity Ratio", 
    "Net Profit Margin", "ROE - Return On Equity"]].mean()

print("Moyennes des ratios financiers par secteur :")
print(sector_means)

# Visualisation des moyennes par secteur
sector_means.plot(kind="bar", figsize=(12, 6))
plt.title("Moyennes des ratios financiers par secteur")
plt.xlabel("Secteur")
plt.ylabel("Valeurs moyennes")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

#%%
# Identifier les valeurs aberrantes par secteur
outlier_summary = []

for sector in sectors:
    sector_data = df[df["Sector"] == sector]
    
    for ratio in ["Current Ratio", "Debt/Equity Ratio", "Net Profit Margin", "ROE - Return On Equity"]:
        Q1 = sector_data[ratio].quantile(0.25)
        Q3 = sector_data[ratio].quantile(0.75)
        IQR = Q3 - Q1
        outliers = sector_data[(sector_data[ratio] < (Q1 - 1.5 * IQR)) | 
                               (sector_data[ratio] > (Q3 + 1.5 * IQR))]
        count_outliers = outliers.shape[0]
        pourcentage = round((count_outliers / sector_data.shape[0]) * 100, 2)

        outlier_summary.append({
            "Sector": sector,
            "Ratio": ratio,
            "Nombre d'outliers": count_outliers,
            "% d'outliers": pourcentage
        })

# Affichage
outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df)

#%%
#Médiane des ratios financiers par secteur
sector_ratios_median = df.groupby('Sector')[["Current Ratio", "Debt/Equity Ratio", "Net Profit Margin", 
        "ROE - Return On Equity"]].median().sort_values(by='Current Ratio', ascending=True)

# Visualisation avec heatmap
sns.heatmap(sector_ratios_median, annot=True, cmap='coolwarm')
plt.title("Médianes des ratios financiers par secteur :")
plt.show()

#%%
#Distribution du ROE par secteur
for sector in sectors:
    print(f"Distribution pour le secteur : {sector}")
    sector_data = df[df["Sector"] == sector]
    
    # Histogramme pour un ratio spécifique
    plt.figure(figsize=(10, 6))
    sns.histplot(sector_data["ROE - Return On Equity"], bins=30, kde=True)
    plt.title(f"Distribution du ROE ({sector})")
    plt.xlabel("ROE - Return On Equity")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()

#%%
#Winsorisation des ratios financiers pour réduire l'impact des valeurs extrêmes
#Importation de la fonction de winsorisation
from scipy.stats.mstats import winsorize

#Copie de df pour la winsorisation
df_winsorized = df.copy()

#Liste des ratios financiers à winsoriser
ratios = ["Current Ratio", "Debt/Equity Ratio", "Net Profit Margin", "ROE - Return On Equity"]

#Myennes par secteur avant winsorisation
sector_means_before = df.groupby("Sector")[ratios].mean()

#Winsorisation (par exemple : on limite les valeurs extrêmes aux 1er et 99e percentiles)
for ratio in ratios:
    lower = 0.05 
    upper = 0.05  
    df_winsorized[ratio] = winsorize(df[ratio], limits=(lower, upper))

#Moyennes par secteur après winsorisation
sector_means_after = df_winsorized.groupby("Sector")[ratios].mean()

#Visualisation des moyennes avant winsorisation
sns.heatmap(sector_means_before, annot=True, cmap="coolwarm")
plt.title("Moyennes des ratios financiers par secteur (Avant Winsorisation)")
plt.show()

#Visualisation des moyennes après winsorisation
sns.heatmap(sector_means_after, annot=True, cmap="coolwarm")
plt.title("Moyennes des ratios financiers par secteur (Après Winsorisation)")
plt.show()

#%%
#SCORE RISQUE 
#%%
#Création du score risque entreprise
#Calcul de la corrélation entre les ratios et la notation
correlations = df[["Rating_Score", "Current Ratio", "Debt/Equity Ratio", 
                   "Net Profit Margin", "ROE - Return On Equity"]].corr()

#Corrélations avec Rating_score 
correlation_rating = correlations["Rating_Score"].drop("Rating_Score")
print(correlation_rating)

#%%
#Poids du ratio sur le rating en valeur absolue pour éviter les effets de signe
weights = correlation_rating.abs() / correlation_rating.abs().sum()

print("Poids des ratios basés sur leur corrélation avec la notation :")
print(weights)

#%%
#Calcul du score de risque pondéré
df["Risk_Score"] = (
    df["Current Ratio"] * weights["Current Ratio"] +
    df["Debt/Equity Ratio"] * weights["Debt/Equity Ratio"] +
    df["Net Profit Margin"] * weights["Net Profit Margin"] +
    df["ROE - Return On Equity"] * weights["ROE - Return On Equity"]
)

# Afficher les premiers scores
print(df[["Corporation", "Risk_Score"]].head())

#%%
df["Risk_Score"].describe()

#%%
#Distribution des scores de risque 
plt.figure(figsize=(10, 6))
sns.histplot(df["Risk_Score"], bins=30, kde=True, color="blue")
plt.title("Distribution des scores de risque")
plt.xlabel("Score de risque")
plt.ylabel("Fréquence")
plt.grid(True)
plt.show()

#%%
#Création du score risque secteur (Normalisé)
#%%
#Importation du Min-Max Scaler pour la normalisation
from sklearn.preprocessing import MinMaxScaler

#Création du scaler
scaler = MinMaxScaler()

#%%
#Calcul des médianes des ratios par secteur
sector_medians = df.groupby("Sector")[ratios].median()

print(sector_medians)

#Normalisation des médianes sectorielles
sector_medians_normalized = sector_medians.copy()
sector_medians_normalized[ratios] = scaler.fit_transform(sector_medians[ratios])

# Vérifier les données normalisées
print(sector_medians_normalized)

#%%
# Calcul de la notation moyenne (médiane) par secteur
sector_medians_normalized["Rating_Score"] = df.groupby("Sector")["Rating_Score"].median()

# Vérifier les notations moyennes par secteur
print(sector_medians_normalized)

#%%
#Corrélations entre les ratios et la notation moyenne
sector_correlations = sector_medians_normalized.corr()["Rating_Score"].drop("Rating_Score")

print(sector_correlations)

# %%
#Poids du ratio sur le rating en valeur absolue
sector_weights = sector_correlations.abs() / sector_correlations.abs().sum()

print("Poids des ratios basés sur leur corrélation avec la notation :")
print(sector_weights)

#%%
#Calcul score de risque pour les secteurs
sector_medians_normalized["Sector_Risk_Score"] = (
    sector_medians_normalized["Current Ratio"] * sector_weights["Current Ratio"] +
    sector_medians_normalized["Debt/Equity Ratio"] * sector_weights["Debt/Equity Ratio"] +
    sector_medians_normalized["Net Profit Margin"] * sector_weights["Net Profit Margin"] +
    sector_medians_normalized["ROE - Return On Equity"] * sector_weights["ROE - Return On Equity"]
).round(3)

print(sector_medians_normalized[["Sector_Risk_Score"]])

# %%
# Visualisation des scores sectoriels
plt.figure(figsize=(12, 6))
sector_medians_normalized["Sector_Risk_Score"].sort_values().plot(kind="bar", color="skyblue")
plt.title("Scores de risque par secteur (basés sur les médianes normalisées)")
plt.xlabel("Secteur")
plt.ylabel("Score de risque")
plt.grid(True)
plt.show()

#%%
#Normalisation des ratios financiers des entreprises
df_normalized = df.copy()
df_normalized[ratios] = scaler.fit_transform(df[ratios])

#Calcul du score de risque pour les entreprises (normalisé)
df_normalized["Risk_Score"] = (
    df_normalized["Current Ratio"] * sector_weights["Current Ratio"] +
    df_normalized["Debt/Equity Ratio"] * sector_weights["Debt/Equity Ratio"] +
    df_normalized["Net Profit Margin"] * sector_weights["Net Profit Margin"] +
    df_normalized["ROE - Return On Equity"] * sector_weights["ROE - Return On Equity"]
).round(3)

print(df_normalized[["Corporation", "Risk_Score"]].head())

#%%
#Comparaison des scores de risque
#Ajout des scores de risque sectoriels à df_normalized
df_normalized = pd.merge(df_normalized, sector_medians_normalized[["Sector_Risk_Score"]], 
                         left_on="Sector", right_index=True)

df_normalized["Risk_Comparaison"] = df_normalized["Risk_Score"] - df_normalized["Sector_Risk_Score"]

# %%
#EXPORT DES DONNEES
#%%
print(df.columns)
print(df_normalized.columns)

# %%
#Sélection des colonnes à exporter
export_colonnes1 = ["Corporation", "Sector", "SubSector", "Industry", 
                   "Rating Agency", "Rating", "Rating_Score","Investment Grade", 
                   "Current Ratio", "Debt/Equity Ratio", 
                   "Net Profit Margin", "ROE - Return On Equity", 
                   "Risk_Score"]

export_colonnes2 = export_colonnes1 + ["Sector_Risk_Score", "Risk_Comparaison"]

#Export des données non normalisées
df[export_colonnes1].to_excel("CorporateCredit_NonNormalized.xlsx", index=False)

#Export des données normalisées
df_normalized[export_colonnes2].to_excel("CorporateCredit_Normalized.xlsx", index=False)

print("Export terminé : fichiers Excel générés.")

# %%
