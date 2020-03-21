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
    status_encoding = {'Alien': 0, 'Android': 1, 'Demon': 2, 'Enhanced Human': 3, 'Entity': 4,
                       'God': 5, 'Human': 6, 'Mutant/Metahuman': 7}
    level_encoding = {'Normal': 0, 'nonspecified': 1, 'Human': 2, 'Athlete': 3, 'Superhuman': 4,
                      'Above Superhuman': 5,'God': 6, 'Omega': 7, 'Beyond Omega': 8}

    comic_df.Property = comic_df.Property.map(property_encoding)
    comic_df.Gender = comic_df.Gender.map(gender_encoding)
    comic_df.Side = comic_df.Side.map(side_encoding)
    comic_df.Status = comic_df.Status.map(status_encoding)
    comic_df.Level = comic_df.Level.map(level_encoding)

    data_encoder = preprocessing.LabelEncoder()
    for column in comic_df.iloc[:, 4:17]:
        comic_df[column] = data_encoder.fit_transform(comic_df[column])

    return comic_df


# ***compare features and determine relationships***
def make_plots(comic_df):
    comic_df.plot(kind='scatter', x='Gender', y='Level', color='purple')
    plt.savefig('plots\\gender_to_Power.png')

    comic_df.plot(kind='scatter', x='Gender', y='Ability to Alter Space/Time/Reality', color='purple')
    plt.savefig('plots\\gender_to_SRT.png')

    comic_df.plot(kind='scatter', x='Property', y='Level', color='purple')
    plt.savefig('plots\\property_to_Power.png')

    comic_df.plot(kind='scatter', x='Side', y='Level', color='purple')
    plt.savefig('plots\\side_to_Power.png')

    comic_df.plot(kind='scatter', x='Status', y='Level', color='blue')
    plt.savefig('plots\\Status_to_Power.png')

    comic_df.plot(kind='scatter', x='Ability to Alter Space/Time/Reality', y='Level', color='blue')
    plt.savefig('plots\\STR_to_Power.png')

    comic_df.plot(kind='scatter', x='Magic/Mystic Powers', y='Level', color='green')
    plt.savefig('plots\\Magic_to_Power.png')

    comic_df.plot(kind='scatter', x='Innate Powers', y='Level', color='green')
    plt.savefig('plots\\Innate_to_Power.png')


# ***create the clustering model***
def make_clusters(comic_df):
    # find the ideal K
    X = comic_df
    distortions = []

    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.plot(range(1, 11), distortions, marker=0)
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig('plots\\ideal_k.png')

    # the result of this plot shows that four is perhaps the ideal number of clusters for this set
    final_k = KMeans(n_clusters=4, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = final_k.fit_predict(X)
    X = X.to_numpy()

    # plot the clusters
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='green', marker='o', edgecolor='black', label='cluster 1')
    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='blue', marker='o', edgecolor='black', label='cluster 2')
    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='purple', marker='o', edgecolor='black', label='cluster 3')
    plt.scatter(X[y_km == 3, 0], X[y_km == 3, 1], s=50, c='red', marker='o', edgecolor='black', label='cluster 4')

    # plot the centroids
    plt.scatter(final_k.cluster_centers_[:, 0], final_k.cluster_centers_[:, 1], s=250, marker='*', c='black',
                edgecolor='black', label='centroids')

    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()


cleaned_comic_df = prepare_data(comic_dataframe)
# make_plots(cleaned_comic_df)
make_clusters(cleaned_comic_df)
print('Process finished.')


