import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics
from sklearn import preprocessing
import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist



## Standardizing data

def multiple_scatters(df):
    '''
    This function takes in a dataframe. If it's a large dataframe, it
    samples 10 random columns and returns pair plots. Otherwise, it
    does pair plots on all the columns.
    '''

    if df.shape[1] > 10:
        df_new = df.sample(10, axis=1)

        sns.set()
        sns.pairplot(df_new, size = 2.5)
    else:
        sns.set()
        sns.pairplot(df, size=2.5)

    plt.show()


def standardizing(df, methods):
    '''
    This function takes in  a dataframe and a method for standardizing, it
    returns the standardized dataframe.

    The methods are:
         - z: for z-scores
         - mm: for min-max
         - robust: for robust
         - gauss: for gaussian
    '''

    if methods == 'z':
        scaler = preprocessing.StandardScaler().fit(df)
        scaled_df = pd.DataFrame(scaler.transform(df))
    elif methods == 'mm':
        scaler = preprocessing.MinMaxScaler().fit(df)
        scaled_df = pd.DataFrame(scaler.transform(df))
    elif methods == 'robust':
        scaler = preprocessing.RobustScaler().fit(df)
        scaled_df = pd.DataFrame(scaler.transform(df))
    else:
        scaler = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
        scaled_df = pd.DataFrame(scaler.fit_transform(df))

    return scaled_df


## Clustering data

def elbow_graph(df, range_clusters):
    '''
    This function takes in a dataframe and range of clusters, and
    determine the elbow graph which would ideally tell us the optimal
    number of clusters to use.
    '''

    distortions = []
    for k in range_clusters:
        kmeanModel = KMeans(n_clusters=k).fit(df)
        kmeanModel.fit(df)
        distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])

    plt.plot(range_clusters, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def plot_clusters(data, algorithm, args, kwds):
    '''
    This function takes in a dataframe, algorithm, arguments, and clusters.
    It returns a plot of the clusters.
    '''

    sns.set_context('poster')
    sns.set_color_codes()
    plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

    if algorithm == 'k':
        start_time = time.time()
        labels = cluster.KMeans(*args, **kwds).fit_predict(data)
        end_time = time.time()
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(data[0], data[1], c=colors, **plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title('Clusters found by {}'.format(str(algorithm)), fontsize=24)

    elif algorithm == 'mean':
        start_time = time.time()
        labels = cluster.MeanShift(*args, **kwds).fit_predict(data)
        end_time = time.time()
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(data[0], data[1], c=colors, **plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title('Clusters found by {}'.format(str(algorithm)), fontsize=24)

    elif algorithm == 'spec':
        start_time = time.time()
        labels = cluster.SpectralClustering(*args, **kwds).fit_predict(data)
        end_time = time.time()
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(data[0], data[1], c=colors, **plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title('Clusters found by {}'.format(str(algorithm)), fontsize=24)

    else:
        start_time = time.time()
        labels = cluster.AgglomerativeClustering(*args, **kwds).fit_predict(data)
        end_time = time.time()
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(data[0], data[1], c=colors, **plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title('Clusters found by {}'.format(str(algorithm)), fontsize=24)
