import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd
import openpyxl

# sklearn libraries
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# ***pull in the data***
comic_dataset = pd.read_excel('datasets\\comic_dataset_1.xlsx')
comic_dataframe = pd.DataFrame(comic_dataset)


# ***clean and prepare the data***
def prepare_data(comic_df):
    # name of specific character is irrelevant
    del comic_df['Name']

    # encode the data with custom mappings
    property_encoding = {'DC': 0, 'Marvel': 1}
    gender_encoding = {'Male': 0, 'Female': 1, 'Neither': 2}
    side_encoding = {'Villain': 0, 'Hero': 1, 'Neither': 2, 'Both': 3, 'Antihero': 4}
    status_encoding = {'Human': 0, 'Enhanced Human': 1, 'Android': 2, 'Alien': 3, 'Demon': 4,
                       'God': 5, 'Mutant/Metahuman': 6, 'Entity': 7}
    immortal_encoding = {'No': 0, 'Semi': 1, 'Yes': 2}
    level_encoding = {'Delta': 0, 'Epsilon': 1, 'Gamma': 2, 'Beta': 3, 'Alpha': 4,
                      'Omega': 5, 'Sigma': 6}

    comic_df.Property = comic_df.Property.map(property_encoding)
    comic_df.Gender = comic_df.Gender.map(gender_encoding)
    comic_df.Side = comic_df.Side.map(side_encoding)
    comic_df.Status = comic_df.Status.map(status_encoding)
    comic_df.Immortality = comic_df.Immortality.map(immortal_encoding)
    comic_df.Level = comic_df.Level.map(level_encoding)

    data_encoder = preprocessing.LabelEncoder()
    for column in comic_df.iloc[:, 5:17]:
        comic_df[column] = data_encoder.fit_transform(comic_df[column])

    return comic_df


# ***add in power scores***
def power_scores(comic_df):
    powerscore_list = []
    for i, j in comic_df.iterrows():
        base_score = ((j.Immortality * 2) + j.Innate_Powers + j.Magic_Or_Mystic_Powers) * (j.Status + 1)
        combat_score = (j.Healing_Factor * 3) + (j.Combat_Skills + j.Use_of_Combative_Weapons) + \
                       (j.Super_Strength + j.Super_Speed)**2
        item_score = (j.Use_of_Enhanced_Tech * 2) + (j.Magical_or_Powerful_Weapons_Items * 3)
        advanced_score = j.Telekinesis_Telepathy + j.Ability_to_Alter_Matter_or_Energy**2 + \
                         j.Ability_to_Alter_Space_Time_Reality**3
        j.Power_Score = (j.Level + 1)**2 * (base_score + combat_score + item_score + advanced_score)
        powerscore_list.append(j.Power_Score)
    comic_df.Power_Score = powerscore_list

    return comic_df


# ***compare features and determine relationships***
def make_plots(comic_df):
    comic_df.plot(kind='scatter', x='Gender', y='Power_Score', color='purple')
    plt.savefig('plots\\gender_to_Power.png')

    comic_df.plot(kind='scatter', x='Side', y='Power_Score', color='purple')
    plt.savefig('plots\\side_to_Power.png')

    comic_df.plot(kind='scatter', x='Status', y='Power_Score', color='blue')
    plt.savefig('plots\\Status_to_Power.png')

    comic_df.plot(kind='scatter', x='Level', y='Power_Score', color='blue')
    plt.savefig('plots\\STR_to_Power.png')

    comic_df.plot(kind='scatter', x='Magic_Or_Mystic_Powers', y='Power_Score', color='green')
    plt.savefig('plots\\Magic_to_Power.png')

    comic_df.plot(kind='scatter', x='Innate_Powers', y='Power_Score', color='green')
    plt.savefig('plots\\Innate_to_Power.png')


# ***create the clustering model***
def create_clusters(comic_df):
    comic_df.to_csv('Datasets\\Updated_Comic_Dataset.csv')

    # find the ideal K
    X = comic_df[5:]
    distortions = []

    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.plot(range(1, 11), distortions, marker=0)
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig('plots\\ideal_k.png')
    plt.show()
    # the result of this plot shows that three is perhaps the ideal number of clusters for this set

    return X


def create_final_model(X_train):
    final_k = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    pred_y = final_k.fit_predict(X_train)
    X_train = X_train.to_numpy()

    # plot the clusters
    plt.scatter(X_train[pred_y == 0, 0], X_train[pred_y == 0, 1], s=50, c='green', marker='o', edgecolor='black', label='cluster 1')
    plt.scatter(X_train[pred_y == 1, 0], X_train[pred_y == 1, 1], s=50, c='blue', marker='o', edgecolor='black', label='cluster 2')
    plt.scatter(X_train[pred_y == 2, 0], X_train[pred_y == 2, 1], s=50, c='purple', marker='o', edgecolor='black', label='cluster 3')

    # plot the centroids
    plt.scatter(final_k.cluster_centers_[:, 0], final_k.cluster_centers_[:, 1], s=250, marker='*', c='black',
                edgecolor='black', label='centroids')

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()


cleaned_comic_df = prepare_data(comic_dataframe)
scored_cleaned_comic_df = power_scores(cleaned_comic_df)
# make_plots(scored_cleaned_comic_df)
X = create_clusters(scored_cleaned_comic_df)
create_final_model(X)

print('Process finished.')


