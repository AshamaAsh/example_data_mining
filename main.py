import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import statistics

def import_file(filename):
    df = pd.read_csv(filename)
    for i in range(106):
        if i < 103:
            col_name = 'expr_level_' + str(i + 1)
            df.columns.values[i] = col_name
        if i > 102 and i < 106:
            col_name = 'spec_func_' + str(i + 1)
            df.columns.values[i] = col_name

    df.columns.values[106] = 'target'

    return df

def cv_decision_tree(x, y, kfold):
    clf = DecisionTreeClassifier(random_state=42)
    k_folds = KFold(n_splits = kfold)
    scores = cross_val_score(clf, x, y, cv = k_folds)
    # print(scores)
#     print('Average CV: ',scores.mean())
    return scores.mean()


def scaler(x, type_scale):
    """ type_scale should be 'MinMax' and 'Standardize' """
    if type_scale == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scale = scaler.fit_transform(x)
    X_scale = pd.DataFrame(scale, index=x.index, columns=x.columns)
    return X_scale

def outlier_detect_knn(df):
    cls_imp = df.copy()
    cls_imp = cls_imp.drop(['target'], axis=1)
    #     cls_imp
    outlier_clean = cls_imp.iloc[:, 0:103]
    col_name = outlier_clean.columns.tolist()
    # col_name
    for col in col_name:
        #     data = outlier_clean.loc[:, [col]]
        clf = NearestNeighbors(n_neighbors=10)

        # fit model
        clf.fit(outlier_clean.loc[:, [col]])
        distances, indexes = clf.kneighbors(outlier_clean.loc[:, [col]])
        #     plt.plot(distances.mean(axis =1))

        distances = pd.DataFrame(distances)
        distances_mean = distances.mean(axis=1)
        # distances_mean  ## mean distance of all data points
        #     print(distances_mean.describe()) # statistical information of distance mean
        std = statistics.stdev(distances_mean)

        # define threshold
        th = distances_mean.mean() + (2 * std)
        outlier_index = np.where(distances_mean > th)
        for i in outlier_index[0].tolist():
            outlier_clean.at[i, col] = None

    df_outlier_na = pd.concat([outlier_clean, cls_imp.iloc[:, 103:106]], axis=1)
    df_outlier_na['target'] = df['target']
    return df_outlier_na

def outlier_test(df):
    outlier_clean = df.iloc[:, 0:103]
    col_name = outlier_clean.columns.tolist()
    # col_name
    for col in col_name:
        #     data = outlier_clean.loc[:, [col]]
        clf = NearestNeighbors(n_neighbors=10)

        # fit model
        clf.fit(outlier_clean.loc[:, [col]])
        distances, indexes = clf.kneighbors(outlier_clean.loc[:, [col]])
        #     plt.plot(distances.mean(axis =1))

        distances = pd.DataFrame(distances)
        distances_mean = distances.mean(axis=1)
        # distances_mean  ## mean distance of all data points
        #     print(distances_mean.describe()) # statistical information of distance mean
        std = statistics.stdev(distances_mean)

        # define threshold
        th = distances_mean.mean() + (2 * std)
        outlier_index = np.where(distances_mean > th)
        for i in outlier_index[0].tolist():
            outlier_clean.at[i, col] = None

    df_outlier_na = pd.concat([outlier_clean, df.iloc[:, 103:106]], axis=1)

    return df_outlier_na

def imputation(df):
    imp_class = df.copy()
    # sepatate dataframe
    cols_numerical = ['target']
    for i in range(103):
        col_name = 'expr_level_' + str(i + 1)
        cols_numerical.append(col_name)
    cols_nominal = ['spec_func_104', 'spec_func_105', 'spec_func_106', 'target']

    df_numerical = df.loc[:, cols_numerical]
    df_nominal = df.loc[:, cols_nominal]

    # mean of the feature’s class-specific values
    # class 0
    df_numerical.loc[df['target'] == 0] = df_numerical.loc[df['target'] == 0].fillna(
        df_numerical.loc[df['target'] == 0].mean())
    df_nominal.loc[df['target'] == 0] = df_nominal.loc[df['target'] == 0].apply(
        lambda x: x.fillna(x.value_counts().index[0]))
    # class 1
    df_numerical.loc[df['target'] == 1] = df_numerical.loc[df['target'] == 1].fillna(
        df_numerical.loc[df['target'] == 1].mean())
    df_nominal.loc[df['target'] == 1] = df_nominal.loc[df['target'] == 1].apply(
        lambda x: x.fillna(x.value_counts().index[0]))

    df_numerical = df_numerical.drop(['target'], axis=1)
    df_imp_class = pd.concat([df_numerical, df_nominal], axis=1, join='inner')
    #     df_imp_class = df_imp_class.loc[:, ~df_imp_class.columns.duplicated()]

    return df_imp_class

def outlier_iqr(df):
    df_2 = df.copy()
    df_2 = df_2.drop(['target'], axis=1)
    df_2_num = df_2.iloc[:,0:103]
    col_name = df_2_num.columns
    for col in col_name:
        Q1 = np.percentile(df_2_num[col], 25, interpolation = 'midpoint')
        Q2 = np.percentile(df_2_num[col], 50, interpolation = 'midpoint')
        Q3 = np.percentile(df_2_num[col], 75, interpolation = 'midpoint')

        IQR = Q3 - Q1
        low_lim = Q1 - 1.5 * IQR
        up_lim = Q3 + 1.5 * IQR

        outlier_index = []
        for index in range(df.shape[0]):
            x = df_2_num.at[index, col]
            if ((x> up_lim) or (x<low_lim)):
                 outlier_index.append(index)

        for i in outlier_index:
            df_2_num.at[i, col] = None

    df_none = pd.concat([df_2_num, df.iloc[:,103:107]], axis=1)
    return df_none


def preprocessing():
    df = import_file("Ecoli.csv")
    for i in range(106):
        if i < 103:
            col_name = 'expr_level_' + str(i + 1)
            df.columns.values[i] = col_name
        if i > 102 and i < 106:
            col_name = 'spec_func_' + str(i + 1)
            df.columns.values[i] = col_name

    df.columns.values[106] = 'target'

    ## class-specific
    df_imp = imputation(df)
    X_cls = df_imp.iloc[:, 0:106]
    y_cls = df_imp['target']


    # print('Avr. CV of class-specific1 imputation: ',cv_decision_tree(X_cls, y_cls, 5))
    # print('Avr. CV of class-specific1 imputation: ', cv_decision_tree(X_cls, y_cls, 5))

    df_out_na = outlier_detect_knn(df_imp)
    # print(df_out_na.isnull().sum())
    # df_out_iqr = outlier_iqr(df_imp)

    df_no_out = imputation(df_out_na)
    X_no_out = df_no_out.iloc[:, 0:106]
    y_no_out = df_no_out['target']
    # print('Avr. CV of no outlier after imputation: ', cv_decision_tree(X_no_out, y_no_out, 5))

    # df_no_out_iqr = imputation(df_out_iqr)
    # X_no_out_iqr = df_no_out_iqr.iloc[:, 0:106]
    # y_no_out_iqr = df_no_out_iqr['target']
    # print('Avr. CV of no outlier IQR: ', cv_decision_tree(X_no_out_iqr, y_no_out_iqr, 5))

    #
    X_minmax = scaler(X_no_out, 'MinMax')
    X_stardard = scaler(X_cls, 'Standardize')
    # print('Average CV of Min max normalization: ',cv_decision_tree(X_minmax, y_cls, 5))
    # print('Average CV of Stadardize normalization: ',cv_decision_tree(X_stardard, y_cls, 5))
    # print('Average CV of no normalization: ',cv_decision_tree(X_cls, y_cls, 5))

    df_pre = pd.concat([X_minmax, y_cls], axis=1)

    return df_pre


def find_decisiontree_best_param(X_train, y_train):
    params = {'criterion': ['gini', 'entropy'],
              'max_depth': list(range(2, 20, 2)),
              'max_leaf_nodes': list(range(2, 100, 2)),
              'min_samples_leaf': [2, 3, 4, 5]}
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=5, verbose=1)
    grid_search_cv.fit(X_train, y_train)

    return grid_search_cv.best_estimator_

def model_decisiontree(x_tr, y_tr, x_ts):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=2, max_leaf_nodes=3, min_samples_leaf=2,
                                 random_state=42)
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_ts)

    return y_pred


def find_kNN_best_param(cv, X_train, y_train):
    params = {'n_neighbors': list(range(1, 30)),
              'leaf_size': list(range(1, 20, 5)),
              'metric': ['euclidean']}
    grid_search_cv = GridSearchCV(KNeighborsClassifier(), params, cv=cv)
    grid_search_cv.fit(X_train, y_train)

    return grid_search_cv.best_estimator_

def model_knn(x_tr, y_tr, x_ts):
    clf = KNeighborsClassifier(leaf_size=1, metric='euclidean', n_neighbors=7)
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_ts)

    return y_pred

def find_randomforest_best_param(X_train, y_train):
    # params = {'bootstrap': [True, False],
    #          'max_depth': list(range(10,120,10)),
    #          'min_samples_leaf': [1, 2, 4],
    #          'min_samples_split': [2, 3,5, 10],
    #          'n_estimators': [200, 400, 600, 800, 1000]}

    random_grid = {"n_estimators": [200, 400, 600, 800, 1000],
                   "criterion": ['gini', 'entropy', 'log_loss'],
                   'max_features': ['log2', 'sqrt'],
                   'max_depth': list(range(10, 110, 10)),
                   'min_samples_split': [2, 3, 5],
                   'min_samples_leaf': [1, 2, 4],
                   'bootstrap': [True, False]}

    rfc = RandomForestClassifier()
    grid_search_cv = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                                        n_jobs=-1, random_state=42)
    grid_search_cv.fit(X_train, y_train)
    # print(grid_search_cv.best_params_)

    return grid_search_cv.best_estimator_

def model_randomforest(x_tr, y_tr, x_ts):
    clf = RandomForestClassifier(bootstrap=False, max_depth=70, n_estimators=200, min_samples_split=3,
                                 min_samples_leaf=1, max_features='sqrt', criterion='gini', random_state=0)
    clf.fit(x_tr, y_tr)
    y_ts_pred = clf.predict(x_ts)
    #     print(clf.get_params)

    return y_ts_pred

def gaussian(x_tr, y_tr, x_ts):
    clf1 = GaussianNB()
    clf1.fit(x_tr, y_tr)
    y_pred_gaussian = clf1.predict(x_ts)

    return y_pred_gaussian
def multinomial(X_train, y_train, X_test):
    clf2 = MultinomialNB()
    clf2.fit(X_train, y_train)
    y_pred_mt = clf2.predict(X_test)

    return y_pred_mt

def complement(X_train, y_train, X_test):
    clf3 = ComplementNB()
    clf3.fit(X_train, y_train)
    y_pred_com = clf3.predict(X_test)

    return y_pred_com


def majority_vote(df):
    maj_vote = []
    for i in range(df.shape[0]):
        vote = df.iloc[i, :].tolist()
        vote_count = Counter(vote)
        top = vote_count.most_common(1)
        maj_vote.append(top[0][0])

    return np.array(maj_vote)

def make_pred_df(y_dt, y_knn, y_rf, y_gas, y_mt, y_com):
# def make_pred_df(y_dt, y_knn, y_rf, y_gas):
    all_predict = {'dt': y_dt,
                   'randomfr': y_rf,
                   'knn': y_knn,
                   'gaussian': y_gas,
                   'multi': y_mt,
                   'y_com': y_com}

    all_predict_df = pd.DataFrame(all_predict)
    # major_vote_predictions = all_predict_df.mode(axis='columns')
    # major_vote_predictions = np.array(major_vote_predictions.iloc[:, -1])

    return all_predict_df

def classification(df):
    X = df.loc[:, df.columns != 'target']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # best_dt = find_decisiontree_best_param(X_train, y_train)
    # best_knn = find_kNN_best_param(5, X_train, y_train)
    # best_rf = find_randomforest_best_param(X_train, y_train)

    y_pred_dt = model_decisiontree(X_train, y_train, X_test)
    y_pred_knn = model_knn(X_train, y_train, X_test)
    y_pred_rf = model_randomforest(X_train, y_train, X_test)
    y_pred_gauss = gaussian(X_train, y_train, X_test)
    ###
    y_pred_mult = multinomial(X_train, y_train, X_test)
    y_pred_comp = complement(X_train, y_train, X_test)

    # print('dt acc:', accuracy_score(y_test, y_pred_dt), 'f1:', f1_score(y_test, y_pred_dt))
    # print('knn acc:', accuracy_score(y_test, y_pred_knn), 'f1:', f1_score(y_test, y_pred_knn))
    # print('rf acc:', accuracy_score(y_test, y_pred_rf), 'f1:', f1_score(y_test, y_pred_rf))
    # print('gaussian acc:', accuracy_score(y_test, y_pred_gauss), 'f1:', f1_score(y_test, y_pred_gauss))
    # ###
    # print('multinomial acc:', accuracy_score(y_test, y_pred_mult), 'f1:', f1_score(y_test, y_pred_mult))
    # print('complement acc:', accuracy_score(y_test, y_pred_comp), 'f1:', f1_score(y_test, y_pred_comp))

    ###
    all_predict = make_pred_df(y_pred_dt, y_pred_knn, y_pred_rf, y_pred_gauss, y_pred_mult, y_pred_comp)
    # all_predict = make_pred_df(y_pred_dt, y_pred_knn, y_pred_rf, y_pred_gauss)
    maj_pred = majority_vote(all_predict)
    # print('maj vote acc:', accuracy_score(y_test, maj_pred), 'f1:', f1_score(y_test, maj_pred))


def main(df):
    X = df.loc[:, df.columns != 'target']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred_knn = model_knn(X_train, y_train, X_test)

    return y_test , y_pred_knn, X_train, y_train

def create_result_file(pred, acc, f1, student_id):
    df1 = pd.DataFrame(data={"pred": pred})
    df1['None'] = np.nan
    df1.loc[len(df1.index)] = [str(round(acc, 3)), round(f1, 3)]
    df1.to_csv(student_id, index=False, header=False)

def run_test(filename, X_train, y_train):
    df = pd.read_csv(filename)
    for i in range(106):
        if i < 103:
            col_name = 'expr_level_' + str(i + 1)
            df.columns.values[i] = col_name
        if i > 102 and i < 106:
            col_name = 'spec_func_' + str(i + 1)
            df.columns.values[i] = col_name
    df_out = outlier_test(df)

    df_out_num = df_out.iloc[:,0:103]
    for i in range(df_out_num.shape[1]):
        df_out_num.iloc[:,i] = df_out_num.iloc[:,i].fillna(df_out_num.iloc[:,i].mean())
    df_imp = pd.concat([df_out_num, df.iloc[:,103:106]], axis=1)
    X_minmax = scaler(df_imp, 'MinMax')
    y_pred_knn = model_knn(X_train, y_train, X_minmax)

    return y_pred_knn


if __name__ == "__main__":
    """ 
    for reproduce test,
    run_test(filename, x_train, y_train)
        filename = file that you want to test
        x_train, y_train = fixed
    create_result_file(predict_y, accuracy_training, f1_training, filename)
        filename = file name that you want to create. 
    """
    ### training
    df_pre = preprocessing() ## use minmax
    y_true , pred_train, X_train, y_train = main(df_pre)
    train_acc = accuracy_score(y_true, pred_train)
    train_f1 = f1_score(y_true, pred_train)
    #### testing
    pred_test = run_test('Ecoli_test.csv', X_train, y_train)
    create_result_file(pred_test,train_acc, train_f1, 's4692549.csv')
