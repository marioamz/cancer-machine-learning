import pandas as pd
import missing_data_code as mdc
import re
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from sklearn import preprocessing, decomposition, cluster, covariance, manifold


def go_explore(excelname, components, index1, index2, index3, buckets, year):
    '''
    This is the go function for exploration. It returns
        - a text file with summary stats
        - a correlation heatmap
        - box plots
        - explained variance graph
        - 2 dimension PCA graph
        - a factor loadings excel
        - a kmeans composite index
        - an affinity propagation index
    '''

    graph_path = 'graphs'
    results_path = 'results'

    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    df_noindex = mdc.reading_in(excelname)
    df_index = df_noindex.set_index([index1, index2])

    # Data exploration functions
    explore_data(df_index)
    explore_potential_correlations(df_index, year)
    box_plot(df_index, year)

    # Principal Component Analysis functions
    pc, y_pca = pca(df_index, components)
    final = pca_df(df_noindex, pc, y_pca, components, index1, index2, year)
    two_dimension_pca_graph(final, index1, year)
    factor_loadings(pc, df_noindex, final, index1, index2)

    # Clustering functions
    kmeans_index(df_noindex, df_index, buckets, [index1, index2, index3])
    affinity_propagation(df_noindex, df_index, [index1, index2, index3])


####CLEAN DATA####

def dummify(df, to_dummy):
    '''
    This function takes a list of columns to turn from categorical
    to dummy variable features
    '''

    new_df = pd.get_dummies(df, columns = to_dummy, dummy_na=True)

    return new_df


def clean_data(df, discretize, bins, binary, to_del):
    '''
    This function cleans the data by removing variables that have
    multicollinearity, filling any nulls, and discretizing/turning
    variables to binary.
        - df = dataframe
        - discretize = a list of variables to discretize
        - binary = a list of variables to turn into binary
        - to_del = a list of variables to be dropped

    '''

    #Discretize
    for var in discretize:
        to_discretize(df, var, bins)

    #Binary
    for binary_var in binary:
        to_binary(df, binary_var)

    #Delete
    for delvar in to_del:
        del df[delvar]

    return df


def to_discretize(df, var, bins):
    '''
    This function discretizes variables into bins that are given
    categorical values based on the bins attribute.
    '''

    bin = binned_statistic(df[var], df[var], bins=bins)[2]

    new_name = var + '_discretize'
    df[new_name] = bin
    del df[var]


def remove_outliers(df, cols):
    '''
    This function removes anything outside of three standard deviations within
    a column
    '''

    df = df[((df[cols] - df[cols].mean()) / df[cols].std()).abs() < 3]
    return df


def to_binary(df, var):
    '''
    This function turns a column of continous variables into binary
    for ease of processing
    '''

    new_col = str(var) + '_binary'
    df[new_col] = (df[var] >= df[var].mean()).astype(int)

    del df[var]


def true_to_false(df, cols):
    '''
    This function takes the dataframe and a list of columns with just
    True or False values, and turns True to 1 and False to 0
    '''

    label_enc = preprocessing.LabelEncoder()

    for c in cols:
        label_enc.fit(df[c])
        new_var = label_enc.transform(df[c])
        new_name = c + '_new'
        df[new_name] = new_var
        del df[c]


####EXPLORATION####

def explore_data(df):
    '''
    This function runs a series of basic exploratory commands, including:
        - Tail and Head
        - Summary Stats
        - Null Values
        - Histograms

    It then calls functions that impute null values, remove instances of
    multicollinearity, and discretize or turn varibales into binary numbers
    '''

    for i in df:
        print('Summary stats for' + ' ' + str(i))
        print(str(df[i].describe()))
        print('\n')

    #Histograms for every column
    plot_data(df, 'hist', None, None)


def explore_potential_correlations(df):
    '''
    This function explores potential correlations between variables

    This is to find instances of multicollinearity.
    '''

    fig4 = plt.figure(figsize=(20,20))
    sns.heatmap(df.corr(), square=True, cmap='PiYG')
    plt.title('Heatmap')
    plt.show()


def count_values(df, col, sort_by, ascending=False):
    '''
    Find the values that make up a particular column of interst
    '''
    groupby = df.groupby(col, sort=False).count()
    return groupby.sort_values(by=sort_by, ascending=False)


#### GRAPHS #####

def box_plot(df, year):
    '''
    This function creates box plots to better identify which columns
    have outliers
    '''

    for col in df:
        #if type(df[col][0]) is not str:
        fig3 = plt.figure(figsize=(8,8))
        plt.boxplot(df[col])
        plt.title('Outliers in' + ' ' + col)
        names = 'graphs/boxplot_' + col + '_' + str(year) + '.png'
        fig3.savefig(names, bbox_inches='tight')
        plt.cla()
    plt.close('all')

# Generic plot function
def plot_data(df, plot_type, var1, var2):
    '''
    This function builds a few simple plots to visualize and start to
    understand the data. Can be called on the cleaned data, or on the
    original dataframe.
    '''

    if plot_type == 'hist':
        print('Histograms of all columns')
        df.hist(figsize=(20,20))
        plt.show()

    elif plot_type == 'bar':
        print('Bar Chart for', var1)
        df[var1].value_counts().plot('bar', figsize=(20,10))
        plt.show()

    elif plot_type == 'scatter':
        print('Scatter plot between', var1, 'and', var2)
        plt.scatter(df[var1], df[var2])
        plt.show()

    elif plot_type == 'line':
        print('Line graph between', var1, 'and', var2)
        df[[var1, var2]].groupby(var1).mean().plot(figsize=(20,10))
        plt.show()


# Pie plots
def plot_pies(values, labels, colors = ['violet', 'yellow']):
    '''
    Plots a pie chart
    '''

    plt.pie(values, labels = labels, shadow=False, colors= colors, startangle=90, autopct='%1.1f%%')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def pie_vals(df, cols1, ordered='projectid', labels = '', colors = ["violet", "yellow", "green", "blue", "orange"]):
    '''
    Pie chart for the top values in any given column
    '''

    df_to_plot = count_values(df, cols1)
    df_to_plot = df_to_plot[ordered]
    if labels == '':
        labels = tuple(df_to_plot.index)

    plot_pies(tuple(df_to_plot), labels, colors)


# Crosstabs
def graph_crosstab(df, col1, col2):
    '''
    Graph crosstab of two discrete variables

    Inputs: Dataframe, column names (strings)
    '''

    pd.crosstab(df[col1], df[col2]).plot(kind='bar')
    plt.title(col2 + " " + "distribution by" + " " + col1)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()


### PRINCIPAL COMPONENT ANALYSIS ###

def pca(df, components):
    '''
    This function takes in a dataframe and the number of principal
    components we want to find in that dataframe, and returns a PCA
    class which is our decomposed dataframe with the number of
    components and an array of all the principal components as
    vectors
    '''

    std_array = preprocessing.StandardScaler().fit_transform(df)
    pc = decomposition.PCA(n_components = components)
    y_pca = pc.fit_transform(std_array)

    return pc, y_pca


def pca_df(df, pc, y_pca, components, var1, var2, name):
    '''
    This function takes in a dataframe without strings, the principal
    components class we created above, the array of vectors, the
    number of components already established, and our target variable(s).

    It returns a  pandas dataframe where the columns are the principal
    component vectors
    '''

    # graphing and printing the explained variance
    explained_variance_graph(pc, name, components)

    # creating the dataframe of PC vectors
    pcaDF = pd.DataFrame(data = y_pca, columns = \
    ['PC' + str(i+1) for i in range(components)])

    # creating final dataframe
    final = pd.concat([pcaDF, df[var1], df[var2]],  axis=1)

    return final


def explained_variance_graph(vectors, year, components):
    '''
    This function takes decomposed vectors and returns
    a cumulative explained variance graph for the number
    of components specified above
    '''

    fig2 = plt.figure(figsize = (8, 8))
    text = 'Variance explained for first' + ' ' + str(components) + \
    ' ' + 'components' + ' ' + 'in' + ' ' + str(year) + '\n' + \
    str(np.cumsum(vectors.explained_variance_ratio_) * 100)

    plt.plot(np.cumsum(vectors.explained_variance_ratio_))
    plt.ylabel('cumulative explained variance')
    plt.title('Explained Variance')
    plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=10)

    name = 'graphs/Explained_Variance_' + str(year) + '.png'
    fig2.savefig(name, bbox_inches='tight')
    plt.close('all')


def two_dimension_pca_graph(df, var1, year):
    '''
    This function takes in a dataframe with principal vectors as
    columns, as well as the variable of interest that we'd like to
    graph which should also ideally be in the df passed in.

    It plots it on a grid graph
    '''

    fig = plt.figure(figsize = (16,16))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 Component PCA for' + ' ' + str(year), fontsize = 20)


    targets = df[var1].unique().tolist()
    colors = sns.color_palette('husl', n_colors=32)  # a list of RGB tuples
    for target, color in zip(targets,colors):
        indicesToKeep = df[var1] == target
        ax.scatter(df.loc[indicesToKeep, 'PC1']
               , df.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 100)
    ax.legend(targets)
    ax.grid()

    name = 'graphs/2DPCA_' + str(year) + '.png'
    fig.savefig(name, bbox_inches='tight')
    plt.close('all')


def factor_loadings(pc, df_vars, df_pca, var1, var2):
    '''
    This function takes in PCA object created by a sklearn
    decomposition, a dataframe that has all the variables,
    a dataframe that has all the principal component vectors,
    target variables, and returns a dataframe that explains
    the factor loadings for each of the principal component
    vectors created
    '''

    loadings_array = pc.components_.T * np.sqrt(pc.explained_variance_)

    loadings = pd.DataFrame(data=loadings_array,
        index= [c for c in df_vars if c != var1 and c != var2],
        columns= [p for p in df_pca if p != var1 and p != var2])

    loads = loadings.style.apply(highlight_vals)

    writer = pd.ExcelWriter('data/loadings.xlsx')
    loads.to_excel(writer)
    writer.save()


def highlight_vals(s):
    '''
    This is a helper function that takes in a dataframe series,
    finds the values in that series larger than a threshold (0.5)
    , and highlights those values.

    I use it here to highlight the variables that explain most of
    the variance within a principal component vector.
    '''

    return ['background-color: yellow' if v >= 0.5 else '' for v in s]


### CLUSTERING ###


def standardizing(df):
    '''
    This function takes in a dataframe that we need to cluster and
    returns it as a standardized array.
    '''

    return preprocessing.StandardScaler().fit_transform(df)


def kmeans_index(df_wtarget, df_wotarget, n_clusters, variables):
    '''
    This function takes in a dataframe with the target variables,
    a dataframe without the target variable, number of clusters,
    and a list of the variables we want to keep in the final df.
    It runs an unsupervised kmeans algorithm to determine how
    the data should be clustered. It returns a dataframe of the
    clusters.
    '''

    array = standardizing(df_wotarget)

    kmeans = cluster.KMeans(n_clusters = n_clusters).fit(array)
    df_wtarget['score'] = kmeans.labels_
    scores = df_wtarget.filter(items=variables)

    writer = pd.ExcelWriter('results/KMeans.xlsx')
    scores.to_excel(writer)
    writer.save()


def affinity_propagation(df_wtarget, df_wotarget, variables):
    '''
    This function takes in a dataframe with our target variable,
    a dataframe without the target variable, and the title of the
    result. It does an affinity propagation analysis and then prints
    out the clusters we've determined.
    '''

    # Learn a graphical structure from the correlations
    edge_model = covariance.GraphLassoCV()

    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery
    X = df_wotarget.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)

    # #############################################################################
    # Cluster using affinity propagation

    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    df_wtarget['score'] = labels + 1
    scores = df_wtarget.filter(items=variables)

    writer = pd.ExcelWriter('results/affinity_propagation.xlsx')
    scores.to_excel(writer)
    writer.save()
