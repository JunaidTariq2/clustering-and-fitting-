# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:24:29 2023

@author: HANZALLAH
"""

#importing all libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from IPython.display import clear_output
from sklearn.cluster import  KMeans
#reading data from an exel sheet
file = pd.read_csv("energy consuption.csv")
features = ['1991','1992','1993','1994','1995']
# dropping any rows which have missing value
file = file.dropna(subset = features)
#assigning data to the new dataframe.
data = file[features].copy()


#Scalling our data
data = ((data - data.min()) / (data.max() - data.min())) * 9 + 1
data.describe()


#initializing random centriods
def random_centroids(data , k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)
centroids = random_centroids(data, 5)


#label each datapoint
def get_labels(data , centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis =1)))
    return distances.idxmin(axis=1)

labels = get_labels(data ,centroids)


def new_centroids(data,labels,k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T


def  plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components =2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait = True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:, 0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:,1])
    plt.show()
    
max_iterations = 100
k = 4
centroids = random_centroids(data , k)
old_centroids = pd.DataFrame()
iteration =1
while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids
    labels = get_labels(data , centroids)
    centroids = new_centroids(data, labels, k)
    plot_clusters(data, labels, centroids, iteration)
    iteration += 1