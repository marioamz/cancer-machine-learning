# Mario Moreno
# Machine Learning in Cancer
# PA 1

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import ml_pipeline as mp
import sklearn
import keras
import jellyfish as jf
from functools import reduce
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2


def read_csv(paths, chunkster):
    '''
    This function reads in the data in chunks given its size. It takes
    as an argument a list of the excel paths and a chunksize.
    '''

    data = []
    for path in paths:
        chunks= []
        txtfilereader = pd.read_csv(path, chunksize=chunkster)
        for chunk in txtfilereader:
            chunks.append(chunk)
        data.append(pd.concat(chunks, ignore_index=True))

    protein_data = data[0]
    all_data = data[1]

    return protein_data, all_data

### K-Fold Cross Validation train_test_split and Machine Learning functions

def classical_ml(df, var, k, grid_size, models_to_run, join_var, feat_selection, no_feats, multiclass):
    '''
    This function takes a dataframe, creates a features and label pair, and
    then, depending on whether it's doing feature selection or not, calls
    the k-fold cross validation function which runs a set of models on the
    date and returns a list of dataframes.

    These are then turned into a report.
    '''

    y = df[var]
    x = df.loc[:, df.columns != var]

    if feat_selection:
        x_feats = feature_selection(x, y, no_feats)
        x_features = x.columns[x_feats]
        print('features selected:', x_features)
        x_final = x[list(x_features)]
        results = k_fold(x_final, y, grid_size, models_to_run, k, multiclass)

    else:
        results = k_fold(x, y, grid_size, models_to_run, k, multiclass)

    report =  gen_report(results, join_var)

    return report


def k_fold(x, y, grid, models, k, multiclass):
    '''
    This function takes in a feature and label pair, as well as grid size,
    models to run, and the number of k-fold cross validations.

    It creates the k-fold test, train splits and runs a magic loop on each,
    returning the results dataframe and appending to a results list which
    is passed on to the function calling this.
    '''

    results = []
    kfold = KFold(k, True, 1)

    for train, test in kfold.split(x):
        x_pretrain, x_pretest = x.loc[train], x.loc[test]
        y_train, y_test = y.loc[train], y.loc[test]
        # standardize
        x_train, x_test = standardizing(x_pretrain, x_pretest)
        # magic loop
        magic_loop = mp.go(x_train, x_test, y_train, y_test, grid, models, multiclass)
        results.append(magic_loop)

    return results


def deep_learning(df, var, k, multiclass):
    '''
    This function takes the k-fold train and test pairs, runs a keras
    deep learning algorithm on them, and returns the average score.
    '''

    y = df[var]
    x = df.loc[:, df.columns != var]

    kerasscore = []
    kerasloss = []

    kfold = KFold(k, True, 1)

    count = 0
    for train, test in kfold.split(x):
        count += 1
        x_pretrain, x_pretest = x.loc[train], x.loc[test]
        y_train, y_test = y.loc[train], y.loc[test]
        # standardize
        x_train, x_test = standardizing(x_pretrain, x_pretest)
        print('This is k-fold:', count)

        if multiclass:
            model = Sequential()
            model.add(Dense(32, activation='relu', input_dim=x_train.shape[1]))
            model.add(Dense(18, activation='softmax'))
            model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
            # Convert labels to categorical one-hot encoding
            y_train = y_train -1
            y_test = y_test - 1
            one_hot_labels = keras.utils.to_categorical(y_train, num_classes=18)
            pred_labels = keras.utils.to_categorical(y_test, num_classes=18)

            # Train the model, iterating on the data in batches of 32 samples
            mod = model.fit(x_train, one_hot_labels, epochs=10, batch_size=32)
            loss, acc = model.evaluate(x_test, pred_labels, batch_size=32)
            kerasscore.append(acc)
            kerasloss.append(loss)
            plot_history(mod)


        else:
            model = Sequential()
            model.add(Dense(32, activation='relu', input_dim=x_train.shape[1]))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])
            mod = model.fit(x_train, y_train, epochs=10, batch_size=32)
            loss, acc = model.evaluate(x_test, y_test, batch_size=128)
            kerasscore.append(acc)
            kerasloss.append(loss)
            plot_history(mod)


    return np.mean(kerasscore), np.mean(kerasloss)


def feature_selection(x, y, no_features):
    '''
    This function runs the best machine learning model determined by the
    magic loop on a smaller subset of features.
    '''

    k_feat = SelectKBest(chi2, no_features).fit(x, y)
    x_feat = k_feat.get_support(indices=True)

    return x_feat

### Standardizing the data

def standardizing(x_train, x_test):
    '''
    This function takes in the training and test pairs, and standardizes
    using the standard scaler
    '''

    scaler_x = preprocessing.StandardScaler().fit(x_train)
    train_x = scaler_x.transform(x_train)
    test_x = scaler_x.transform(x_test)

    return train_x, test_x

## Generating the report

def gen_report(result_list, join_var):
    '''
    This function takes in a list of dataframes and calculates averages
    for all of them
    '''

    merged_df = reduce(lambda x, y: pd.merge(x, y, on = join_var), result_list)
    merged_cols = list(merged_df.columns)

    acc = []
    auc = []
    f1 = []
    confusion = []

    for col in merged_cols:
        if jf.jaro_winkler('accuracy', col) > 0.9:
            acc.append(col)
        elif jf.jaro_winkler('auc_roc', col) > 0.7:
            auc.append(col)
        elif jf.jaro_winkler('f1_at_5', col) > 0.7:
            f1.append(col)
        elif jf.jaro_winkler('confusion', col) > 0.9:
            confusion.append(col)

    merged_df['acc_mean'] = merged_df[acc].mean(axis=1)
    merged_df['auc_mean'] = merged_df[auc].mean(axis=1)
    merged_df['f1_mean'] = merged_df[f1].mean(axis=1)
    merged_df['conf_sum'] = merged_df[confusion].sum(axis=1)

    report_df = merged_df[[join_var, 'acc_mean', 'auc_mean', 'f1_mean', 'conf_sum']]

    return report_df


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


## EXPLORATORY ANALYSIS
### Find any missing columns to impute later on
### Find any outliers that need to be accounted for
### Graph a few relationships and other things


def find_missing_cols(df):
    '''
    This function takes a dataframe and finds the columns with
    missing variables

    It returns a list of all columns with missing variables
    '''

    null_cols = []
    # creating a series that counts missing values in column
    null_series = df.isnull().sum()

    for ind, val in null_series.items():
        if val != 0:
            null_cols.append(ind)

    return null_cols


def random_columns(df, var, k_columns):
    '''
    Given non-trivially large dataframes, this function randomly samples
    k_columns from the dataframe and creates a smaller one that I can efficiently
    run exploratory analysis on.
    '''

    return pd.concat([df[var], df.sample(k_columns, axis=1)], axis=1)


def bigdata_heatmap(df, var, k_columns):
    '''
    This function takes a dataframe, a variable of interest, and
    a number of columns to find the correlation between the variable
    of interest and k_most correlated columns in the dataset
    '''

    corrmat = df.corr()
    cols = corrmat.nlargest(k_columns, var)[var].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', \
    annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    print('Ten most correlated columns are:', list(cols))
    return list(cols)


def multiple_scatters(df, cols):
    '''
    This function takes in a dataframe and a list of columns, it maps out
    all the scatter plots between that list of columns pretty awesomely.
    '''

    sns.set()
    sns.pairplot(df[cols], size = 2.5)
    plt.show();


def histogram(df, column):
    '''
    This function graphs the histogram for one column in our data
    '''

    return sns.distplot(df[column])


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
