import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.io import arff
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.datasets import make_moons, make_circles, make_classification, make_multilabel_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import time
import xlrd

training_sizes=np.linspace(0.1, 1.0, 10)
acc_train_mean_comb = []
acc_test_mean_comb = []
time_train_mean_comb = []
time_test_mean_comb = []
val_train_mean_comb = []
val_test_mean_comb = []
val_train_mean_comb2 = []
val_test_mean_comb2 = []
fit_time_mean_comb = []
score_time_mean_comb = []
fit_mean_comb = []
score_mean_comb = [] 
acc_test_mean_comb_all = []
fit_train_times_comb_all = []


#ds1 = [0.8788461538461538,0.9596153846153846,0.9740384615384615,0.8932692307692308,0.9740384615384615]
#ds2 = [0.6942307692307692,0.9576923076923077,0.9865384615384616,0.675,0.9913461538461539]
ds1 = [0.872, 0.955, 0.829, 0.912, 0.956]
ds2 = [0.582, 0.916, 0.938, 0.708, 0.927]

def my_barplot(ds, data_set="ds1"):
    my_labels = ['DT', 'KNN', 'SVC', 'ADA', 'ANN']
    my_colors = ['red','green', 'blue', 'maroon', 'cyan']

      
    # creating the bar plot
    fig = plt.figure(figsize = (4, 4))
    plt.barh(my_labels,ds, color=my_colors)
    plt.title('Test Acuracy Scores for Classifiers with best parameters')
    plt.ylabel('Classifiers')
    plt.xlabel('Accuracy')
    plt.show()
    fname="acc_barplot_" + data_set + ".png"
    plt.savefig(fname)
    plt.close()


def test_default(X_train, X_test, y_train, y_test, data_set="dataset1"):
    print("Running DecisionTreeClassifier")
#DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf = DecisionTreeClassifier(max_depth=15)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:", data_set,"Decision Tree", metrics.accuracy_score(y_test, y_pred))
    score = clf.score(X_test, y_test)
    print("DT number of nodes =", clf.tree_.node_count)
    print("DT number of leaf =", clf.tree_.max_depth)
    return
#MLPClassifier
    print("Running calc_accuracy MLPClassifier")
    #print accuracy
    clf = MLPClassifier()
    clf = MLPClassifier(alpha=1, max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:", data_set,"Neural Network", metrics.accuracy_score(y_test, y_pred))

#AdaBoostClassifier
    print("Running calc_accuracy AdaBoostClassifier")
    #PRINT ACCURACY
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:", data_set,"AdaBoosing", metrics.accuracy_score(y_test, y_pred))

#SVM
    print("Running calc_accuracy SVMClassifier")
    #PRINT ACCURACY
    #clf = SVC()
    clf = SVC(gamma=2, C=1)
    #clf = SVC(kernel="linear", C=0.025)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:", data_set,"Support Vector Machine", metrics.accuracy_score(y_test, y_pred))

#KNeighborsClassifier
    print("Running calc_accuracy KNeighborsClassifier")
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:",data_set, "KNeighborsClassifier", metrics.accuracy_score(y_test, y_pred))


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1="name_param_1", name_param_2="name_param_2"):

    scores_mean = cv_results['mean_test_score']

    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']

    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores

    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)

    for idx, val in enumerate(grid_param_2):

        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')

    ax.set_xlabel(name_param_1, fontsize=16)

    ax.set_ylabel('CV Average Score', fontsize=16)

    ax.legend(loc="best", fontsize=15)

    ax.grid('on')
    plt.show()


def print_df(x):
    for i in range(len(x)):
        print(x[i])

def my_plot(x1, y1, x2, y2, title="Title", x_label='x_value', y_label='y_value', legend=['train', 'test'], fname="graph.png", data_set="data_set"):
    title = title + " - " + data_set
    #leg = plt.legend()
    #leg.set_alpha(1)
    colors= ['green', 'palegreen', 'blue', 'lightskyblue']
    colors= ['brown', 'orange', 'blue', 'lightskyblue']
    if data_set == "dataset1":
        plt.plot(x1,y1, color=colors[0], marker='o',label="Train", linestyle='solid')
        plt.plot(x2,y2, color=colors[1], marker='2',label="Test", linestyle='solid')
    else:
        plt.plot(x1,y1, color=colors[2], marker='o',label="Train", linestyle='solid')
        plt.plot(x2,y2, color=colors[3], marker='o',label="Test", linestyle='solid')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(legend)
    #plt.show()
    plt.savefig(fname)
    plt.close()

def my_plot_comb(x, y1, y2, title="Title", x_label='x_value', y_label='y_value', legend=['train_dataset1', 'test_dataset1', 'train_dataset2', 'test_dataset2'], fname="graph_comb.png"):
    colors= ['green', 'palegreen', 'blue', 'lightskyblue']
    colors= ['brown', 'orange', 'blue', 'lightskyblue']
    plt.plot(x,y1[0], color=colors[0], marker='o',label="Train1", linestyle='solid')
    plt.plot(x,y2[0], color=colors[1], marker='2',label="Test1", linestyle='solid')
    plt.plot(x,y1[1], color=colors[2], marker='1',label="Train2", linestyle='solid')
    plt.plot(x,y2[1], color=colors[3], marker='3',label="Test2",linestyle='solid')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(legend)
    #plt.show()
    plt.savefig(fname)
    plt.close()

#    my_plot_comb(training_sizes, acc_test_mean_comb_all, title="All Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname="acc_comb_all_dt.png")
def my_plot_comb_all(x, y1, title="Title", x_label='x_value', y_label='y_value', legend=['dt', 'ann', 'ada', 'svm', 'knn'], fname="graph_comb_all.png"):
    print("y1[0]=", y1[0])
    print("y1=", y1[1])
    print("y1[1][0]=", y1[1][0])

    plt.plot(x,y1[0], color='blue', marker='o',label="Test1", linestyle='solid')
    plt.plot(x,y1[2], color='green', marker='o',label="Test1", linestyle='solid')
    plt.plot(x,y1[4], color='red', marker='o',label="Test1", linestyle='solid')
    plt.plot(x,y1[6], color='cyan', marker='o',label="Test1", linestyle='solid')
    plt.plot(x,y1[8], color='orange', marker='o',label="Test1", linestyle='solid')
    title="All Test Learning Curve - dataset1"
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(legend)
    #plt.show()
    fname1 = "f1_" + fname
    plt.savefig(fname1)
    plt.close()
    #plt.plot(x,y1[1], color='blue', marker='o',label="Test1", linestyle='solid')
    plt.plot(x,y1[1], color='blue', marker='o',label="Test1", linestyle='solid')
    plt.plot(x,y1[3], color='green', marker='o',label="Test1", linestyle='solid')
    plt.plot(x,y1[5], color='red', marker='o',label="Test1", linestyle='solid')
    plt.plot(x,y1[7], color='cyan', marker='o',label="Test1", linestyle='solid')
    plt.plot(x,y1[9], color='orange', marker='o',label="Test1", linestyle='solid')
    title="All Test Learning Curve - dataset2"
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(legend)
    #plt.show()
    fname2 = "f2_" + fname
    plt.savefig(fname2)
    plt.close()

def calc_accuracy_with_bp(X_train, X_test, y_train, y_test, data_set="dataset1"):
    print("Running DecisionTreeClassifier")
#DecisionTreeClassifier
    print("calculating accuracy stat for dataset:", data_set)

    #PRINT ACCURACY
    cls_all = ["dt", "ann", "ada", "svm", "knn"]
    for cls in cls_all:
        if data_set == "dataset1" and cls == "dt": 
            print("Running DecisionTreeClassifier with:", data_set)
            clf = DecisionTreeClassifier(max_depth=8,  min_samples_leaf= 8, criterion=  'gini')
        elif data_set == "dataset2" and cls == "dt": 
            print("Running DecisionTreeClassifier with:", data_set)
            clf = DecisionTreeClassifier(max_depth=12, criterion = 'gini',min_samples_leaf= 10)
        elif data_set == "dataset1" and cls == "ada": 
            print("Running AdaBoostClassifier with:", data_set)
            clf = AdaBoostClassifier(learning_rate = 0.5, n_estimators= 60)
        elif data_set == "dataset2" and cls == "ada": 
            print("Running AdaBoostClassifier with:", data_set)
            clf = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = 0.5, n_estimators= 200)
        elif data_set == "dataset1" and cls == "svm": 
            print("Running SVC with:", data_set)
            clf = SVC(kernel= 'rbf', max_iter= 500)
        elif data_set == "dataset2" and cls == "svm": 
            print("Running SVC with:", data_set)
            clf = SVC(kernel= 'rbf', max_iter= 10000)
        elif data_set == "dataset1" and cls == "ann": 
            print("Running MLPClassifier with:", data_set)
            #clf = MLPClassifier(activation= 'tanh', hidden_layer_sizes= 20, learning_rate= 'constant', max_iter= 750, solver= 'sgd')
            clf = MLPClassifier(activation= 'tanh', hidden_layer_sizes= 20, learning_rate= 'constant', max_iter= 500, solver= 'sgd')
        elif data_set == "dataset2" and cls == "ann": 
            print("Running MLPClassifier with:", data_set)
            clf = MLPClassifier(activation= 'tanh', hidden_layer_sizes= 20, learning_rate= 'constant', max_iter= 500, solver= 'sgd')

        elif data_set == "dataset1" and cls == "knn": 
            print("Running KNeighborsClassifier with:", data_set)
            clf = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 15, p= 2)
        elif data_set == "dataset2" and cls == "knn": 
            print("Running KNeighborsClassifier with:", data_set)
            clf = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 30, p= 2)

        start_time1 = time.time()
        clf.fit(X_train, y_train)
        end_time1 = time.time()
        start_time2 = time.time()
        y_pred = clf.predict(X_test)
        end_time2 = time.time()
        fit_time=end_time1 - start_time1
        query_time=end_time2 - start_time2
        print("With_best_Parameter_Accuracy", cls, data_set, metrics.accuracy_score(y_test, y_pred), fit_time, query_time)
#        print("Time_to_fit", data_set, cls, fit_time)
#        print("Time_to_query", data_set, cls, query_time)
    #clf = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = 0.5, n_estimators= 100)
#        elif cls == "ada":
        print("F1_scores:", cls, metrics.f1_score(y_test, y_pred, average='macro'))

def calc_accuracy_all(X_train, X_test, y_train, y_test, data_set="dataset1"):
#MLPClassifier
    print("Running calc_accuracy MLPClassifier")
    #print accuracy
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:", data_set,"Neural Network", metrics.accuracy_score(y_test, y_pred))

    ########### run with best parameters
    #clf = MLPClassifier(activation= 'relu', hidden_layer_sizes= 25, learning_rate= 'adaptive', max_iter= 200, solver= 'adam')
    #Best: 0.976250 using {'activation': 'relu', 'hidden_layer_sizes': 200, 'learning_rate': 'constant', 'max_iter': 50, 'solver': 'adam'}
    clf = MLPClassifier(activation= 'relu', hidden_layer_sizes= 25, learning_rate= 'adaptive', max_iter= 200, solver= 'adam')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter MLPClassifier - Accuracy:", data_set,"Neural Network", metrics.accuracy_score(y_test, y_pred))

    #clf = MLPClassifier(activation= 'tanh', hidden_layer_sizes= 200, learning_rate= 'adaptive', max_iter= 200, solver= 'adam')
    #Best: 0.740625 using {'activation': 'tanh', 'hidden_layer_sizes': 200, 'learning_rate': 'adaptive', 'max_iter': 750, 'solver': 'adam'}
    clf = MLPClassifier(activation= 'tanh', hidden_layer_sizes= 200, learning_rate= 'adaptive', max_iter= 750, solver= 'adam')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter MLPClassifier - Accuracy:", data_set,"Neural Network", metrics.accuracy_score(y_test, y_pred))

#AdaBoostClassifier
    print("Running calc_accuracy AdaBoostClassifier")
    #PRINT ACCURACY
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:", data_set,"AdaBoosing", metrics.accuracy_score(y_test, y_pred))

########### run with best parameters
    #clf = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = 0.5, n_estimators= 100)
    #Best: 0.943750 using {'algorithm': 'SAMME', 'learning_rate': 0.7, 'n_estimators': 70}

    clf = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = 0.5, n_estimators= 100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter AdaBoostClassifier - Accuracy:", data_set,"AdaBoosing", metrics.accuracy_score(y_test, y_pred))

    #Best: 0.622917 using {'algorithm': 'SAMME', 'learning_rate': 0.5, 'n_estimators': 20}
    clf = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = 0.5, n_estimators= 20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter AdaBoostClassifier - Accuracy:", data_set,"AdaBoosing", metrics.accuracy_score(y_test, y_pred))

#SVM
    print("Running calc_accuracy SVMClassifier")
    #PRINT ACCURACY
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:", data_set,"Support Vector Machine", metrics.accuracy_score(y_test, y_pred))

    ########### run with best parameters
    #clf = SVC(C= 0.75, degree= 2, kernel= 'rbf', max_iter= 200)
    #Best: 0.977500 using {'C': 0.5, 'degree': 2, 'kernel': 'rbf', 'max_iter': 100}
    clf = SVC(C= 0.5, degree= 2, kernel= 'rbf', max_iter= 100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter SVC - Accuracy:", data_set,"Support Vector Machine", metrics.accuracy_score(y_test, y_pred))

    #clf = SVC(C= 0.75, degree= 2, kernel= 'rbf', max_iter= 200)
    #Best: 0.723958 using {'C': 1.0, 'degree': 2, 'kernel': 'rbf', 'max_iter': 1000}
    clf = SVC(C= 1.00, degree= 2, kernel= 'rbf', max_iter= 1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter - Accuracy:", data_set,"Support Vector Machine", metrics.accuracy_score(y_test, y_pred))

#KNeighborsClassifier
    print("Running calc_accuracy KNeighborsClassifier")
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:",data_set, "KNeighborsClassifier", metrics.accuracy_score(y_test, y_pred))

    ########## run with best parameters
    #clf = KNeighborsClassifier(algorithm= 'auto', leaf_size= 1, n_neighbors= 7, p= 2)
    #Best: 0.970000 using {'algorithm': 'auto', 'n_neighbors': 40, 'p': 2}
    clf = KNeighborsClassifier(algorithm= 'auto', leaf_size= 1, n_neighbors= 40, p= 2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter - Accuracy:",dataset, "KNeighborsClassifier", etrics.accuracy_score(y_test, y_pred))

    #clf = KNeighborsClassifier(algorithm= 'auto', leaf_size= 1, n_neighbors= 7, p= 2)
    #Best: 0.701042 using {'algorithm': 'auto', 'n_neighbors': 7, 'p': 1}
    clf = KNeighborsClassifier(algorithm= 'auto', leaf_size= 1, n_neighbors= 7, p= 1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter KNeighborsClassifier - Accuracy:",metrics.accuracy_score(y_test, y_pred))


def find_best_params(X_train, X_test, y_train, y_test, data_set="data_set", la="dt"):
    kfold = KFold(n_splits=10 )

    #la = "knn"
    param_grid = {}
    learn_algo = DecisionTreeClassifier()
    if la == "dt":
        print("Running DecisionTreeClassifier Exteriment")
        learn_algo = DecisionTreeClassifier()
        criterion = ['gini', 'entropy']
        #max_depth=[6, 7,8, 10, 12]
        max_depth=[1, 2, 3, 4, 5,6, 7,8, 10, 12,15, 20, 25]
        min_samples_leaf = [1,2,3,4,5,6, 8, 10, 20, 30, 40]
        param_grid = dict(max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf)
    if la == "ada":
        print("Running AdaBoostClassifier Exteriment")
        learn_algo = AdaBoostClassifier()
        algorithm = ['SAMME', 'SAMME.R']
        n_estimators = [10, 20, 30, 40, 50, 60, 70, 80, 100]
        learning_rate = [0.1, 0.5, 0.7, 1.0]
        param_grid = dict(n_estimators=n_estimators, algorithm=algorithm, learning_rate=learning_rate)
    if la == "svm":
        print("Running SVC Exteriment")
        learn_algo = SVC()
        kernel=['linear', 'poly', 'rbf', 'sigmoid']
        C = [0.1, 0.25, 0.5, 0.75, 1.0]
        degree = [1, 2, 3, 4,5]
        max_iter = [100, 200, 300, 400, 500, 1000]
        param_grid = dict(degree=degree, kernel=kernel, C=C, max_iter=max_iter)
    if la == "ann":
        print("Running MLPClassifier Exteriment")
        learn_algo = MLPClassifier()
        activation = ['identity', 'logistic', 'tanh', 'relu']
        solver = ['lbfgs', 'sgd', 'adam']
        learning_rate = ['constant', 'invscaling', 'adaptive']

        max_iter = [50, 100, 200, 250, 300, 500, 750]
        hidden_layer_sizes = [20, 30, 40, 50, 100]
        learning_rate_init = [ 0.001, 0.005, 0.01, 0.05]
        #max_iter = [50, 100, 200, 300, 400, 500]
        #hidden_layer_sizes = [25, 50, 100, 150, 200]
        #learning_rate_init = [ 0.001, 0.005]
        param_grid = dict(activation=activation, hidden_layer_sizes=hidden_layer_sizes, solver=solver, learning_rate=learning_rate, max_iter=max_iter)

    if la == "knn":
        print("Running KNeighborsClassifier Exteriment")
        learn_algo = KNeighborsClassifier()
        n_neighbors = [5, 7, 10, 12, 15, 20, 25, 30, 40, 50]
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        #leaf_size = [1, 2, 5, 7, 10, 20, 50, 100]
        p = [ 1, 2]
        #param_grid = dict(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, p=p)
        param_grid = dict(n_neighbors=n_neighbors, algorithm=algorithm, p=p)

    print("######## Normal cross_val_score:######## ")
    result = cross_val_score(learn_algo, X_train, y_train, cv=kfold, scoring='accuracy')
    print("mean_score: ", result.mean())
    print("Normal cross_val_score result=", result)

    #param_grid = dict(max_depth=max_depth, criterion=criterion)

    print("######## GridSearchCV cross_val_score:######## ")
    grid = GridSearchCV(estimator=learn_algo, param_grid=param_grid, cv = 10, n_jobs=-1, return_train_score=True)

    start_time = time.time()
    grid_result = grid.fit(X_train, y_train)
    #print(grid_result)
    # Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print("Execution time: " + str((time.time() - start_time)) + ' ms')
    print("mean_score: ", result.mean())
    #print(grid.cv_results_)
    #print("grid_result.grid_scores=", grid_result.grid_scores_)

    print("######## RandomizedSearchCV cross_val_score:######## ")
    random = RandomizedSearchCV(estimator=learn_algo, param_distributions=param_grid, cv = 10, n_jobs=-1)

    start_time = time.time()
    random_result = random.fit(X, y)
    # Summarize results
    print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
    print("Execution time: " + str((time.time() - start_time)) + ' ms')
    print("mean_score: ", result.mean())
    print("result=", result)
 
    #plot_grid_search(result, "grid_param_1", "grid_param_2", "name_param_1", "name_param_2")
    #exit(0)
def experiment_dt(X_train, X_test, y_train, y_test, data_set="data_set"):

    global train_mean_comb
    global test_mean_comb 

    #ACCURACY and TIME
    training_sizes=np.linspace(0.1, 1.0, 10)
    start_time = time.time()
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(DecisionTreeClassifier(max_depth=10), X=X_train, y=y_train, scoring='accuracy', cv=10, n_jobs=-1, return_times=True, train_sizes=training_sizes)
    end_time = time.time()
    print("DT Training time=", end_time - start_time)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    acc_train_mean_comb.append(train_mean)
    acc_test_mean_comb.append(test_mean)
    acc_test_mean_comb_all.append(test_mean)
    fit_train_mean = np.mean(fit_times, axis=1)
    fit_train_times_comb_all.append(fit_train_mean)
    print("acc_test_mean_comb_all=", acc_test_mean_comb_all)
    file_name = "acc_" + data_set + "_dt.png"
    my_plot(training_sizes, train_mean, training_sizes, test_mean, title="Decision Tree Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname=file_name, data_set=data_set)
    fit_mean = np.mean(fit_times, axis=1)
    score_mean = np.mean(score_times, axis=1)
    fit_mean_comb.append(fit_mean)
    time_train_mean_comb.append(fit_mean)
    time_test_mean_comb.append(score_mean)
    score_mean_comb.append(score_mean)
    file_name = "times_" + data_set + "_dt.png"
    my_plot(training_sizes, fit_mean, training_sizes, score_mean, title="Decision Tree Learning Curve Times", x_label="% of Training Size", y_label="Time", fname=file_name, data_set=data_set)
    print("train_sizes=", train_sizes)
    print("train_scores=", train_scores)
    print("test_scores=", test_scores)

    #VALIDATION
    param_range = np.arange(1, 50, 2)
    train_scores, test_scores = validation_curve(DecisionTreeClassifier(), X_train, y_train, param_name="max_depth", param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_train_mean_comb.append(train_mean)
    val_test_mean_comb.append(test_mean)
    file_name = "val_" + data_set + "_dt.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="Validation Curve With Decision Tree", x_label="Depth Of Tree", y_label="Accuracy", fname=file_name, data_set=data_set)
    train_scores, test_scores = validation_curve(DecisionTreeClassifier(), X_train, y_train, param_name="criterion", param_range=["gini", "entropy"], cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    param_range=["gini", "entropy"]
    file_name = "gini_entropy" + data_set + "_dt.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="Decision Tree Cross-Validation score", x_label="% of Training score", y_label="Accuracy", fname=file_name, data_set=data_set)

    # number of leaf
    param_range = np.arange(1, 50, 2)
    train_scores, test_scores = validation_curve(DecisionTreeClassifier(), X_train, y_train, param_name="min_samples_leaf", param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_train_mean_comb2.append(train_mean)
    val_test_mean_comb2.append(test_mean)
    file_name = "val2_" + data_set + "_dt.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="Validation Curve With Decision Tree min_samples_leaf", x_label="min_samples_leaf", y_label="Accuracy", fname=file_name, data_set=data_set)


def experiment_ann(X_train, X_test, y_train, y_test, data_set="data_set"):
#    y_test=y_test.ravel().reshape(-1)
#    y_train=y_train.ravel().reshape(-1)
    training_sizes=np.linspace(0.1, 1.0, 10)

    #ACCURACY and TIMES
    start_time = time.time()
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(MLPClassifier(), X=X_train, y=y_train, scoring='accuracy', cv=10, n_jobs=-1, return_times=True, train_sizes=training_sizes)
    end_time = time.time()
    print("ANN Training time=", end_time - start_time)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    acc_train_mean_comb.append(train_mean)
    acc_test_mean_comb.append(test_mean)
    acc_test_mean_comb_all.append(test_mean)
    fit_train_mean = np.mean(fit_times, axis=1)
    fit_train_times_comb_all.append(fit_train_mean)
    print("acc_test_mean_comb_all=", acc_test_mean_comb_all)
    file_name = "acc_" + data_set + "_ann.png"
    my_plot(training_sizes, train_mean, training_sizes, test_mean, title="ANN Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname=file_name, data_set=data_set)
    fit_mean = np.mean(fit_times, axis=1)
    score_mean = np.mean(score_times, axis=1)
    fit_mean_comb.append(fit_mean)
    score_mean_comb.append(score_mean)
    time_train_mean_comb.append(fit_mean)
    time_test_mean_comb.append(score_mean)
    file_name = "times_" + data_set + "_ann.png"
    my_plot(training_sizes, fit_mean, training_sizes, score_mean, title="ANN Learning Curve Times", x_label="% of Training Size", y_label="Time", fname=file_name, data_set=data_set)
#######
    param_range = np.arange(200, 1000, 200)

    train_scores, test_scores = validation_curve(MLPClassifier(), X_train, y_train, param_name="max_iter", param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
#    val_train_mean_comb.append(train_mean)
#    val_test_mean_comb.append(test_mean)
    file_name = "val_max_iter" + data_set + "_ann.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="ANN Validation Curve", x_label="Maximum Iteration", y_label="Accuracy", fname=file_name, data_set=data_set)
    #exit(1)
#######
    #VALIDATIONS
    #hidden_layer_sizes
    #param_range = np.array([200, 300, 500,700, 1000, 1200, 1500, 1700, 2000, 2200])
    #param_range = np.array([10,20,30, 50, 100, 150, 200, 500])
    #param_range = np.array([(10,10), (20,20),(25,25),(30,30), (50,50), (100,100), (150,150))
    #param_range = np.array([10, 15, 20, 25, 30,40, 50, 100])
    param_range = [(10,10), (20,20),(25,25),(30,30), (50,50), (100,100), (150,150)]
    #param_range = np.array([(25,25,25), (50, 50,50),(75, 75, 75), (100, 100, 100), (150,150,150), (200,200, 200)]
    train_scores, test_scores = validation_curve(MLPClassifier(), X_train, y_train, param_name="hidden_layer_sizes", param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_train_mean_comb2.append(train_mean)
    val_test_mean_comb2.append(test_mean)
    file_name = "val_" + data_set + "_ann.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="ANN Cross-validation score: hidden_layer_sizes", x_label="hidden_layer_sizes", y_label="Accuracy", fname=file_name, data_set=data_set)

#    #alpha
##    param_range = np.array([.00001, .0001, .001, .01])
#    train_scores, test_scores = validation_curve(MLPClassifier(), X_train, y_train, param_name="alpha", param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)
#    train_mean = np.mean(train_scores, axis=1)
#    test_mean = np.mean(test_scores, axis=1)
#    file_name = "val_" + data_set + "_ann.png"
#    my_plot(param_range, train_mean, param_range, test_mean, title="ANN Cross-validation score", x_label="Alpha", y_label="Accuracy", fname=file_name, data_set=data_set)

    #Activation
    param_range = ["identity", "logistic", "tanh", "relu"]
    train_scores, test_scores = validation_curve(MLPClassifier(), X_train, y_train, param_name="activation", param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_train_mean_comb.append(train_mean)
    val_test_mean_comb.append(test_mean)
#    val_train_mean_comb.append(train_mean)
#    val_test_mean_comb.append(test_mean)
    file_name = "val_activ" + data_set + "_ann.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="ANN Cross-Validation score", x_label="Activation", y_label="Accuracy", fname=file_name, data_set=data_set)

    #print accuracy
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:", data_set,"Neural Network", metrics.accuracy_score(y_test, y_pred))

########### run with best parameters
    clf = MLPClassifier(activation= 'relu', hidden_layer_sizes= 25, learning_rate= 'adaptive', max_iter= 200, solver= 'adam')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter - Accuracy:", data_set,"Neural Network", metrics.accuracy_score(y_test, y_pred))

    clf = MLPClassifier(activation= 'tanh', hidden_layer_sizes= 200, learning_rate= 'adaptive', max_iter= 200, solver= 'adam')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter - Accuracy:", data_set,"Neural Network", metrics.accuracy_score(y_test, y_pred))


def experiment_ada(X_train, X_test, y_train, y_test, data_set="data_set"):
    training_sizes=np.linspace(0.1, 1.0, 10)
    #ACCURACY and TIMES
    start_time = time.time()
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(AdaBoostClassifier(), X=X_train, y=y_train, scoring='accuracy', cv=10, n_jobs=1, return_times=True, train_sizes=training_sizes)
    end_time = time.time()
    print("ADA Training time=", end_time - start_time)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    acc_train_mean_comb.append(train_mean)
    acc_test_mean_comb.append(test_mean)
    acc_test_mean_comb_all.append(test_mean)
    fit_train_mean = np.mean(fit_times, axis=1)
    fit_train_times_comb_all.append(fit_train_mean)
    print("acc_test_mean_comb_all=", acc_test_mean_comb_all)
    file_name = "acc_" + data_set + "_ada.png"
    my_plot(training_sizes, train_mean, training_sizes, test_mean, title="AdaBoosting Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname=file_name, data_set=data_set)
    fit_mean = np.mean(fit_times, axis=1)
    score_mean = np.mean(score_times, axis=1)
    fit_mean_comb.append(fit_mean)
    score_mean_comb.append(score_mean)
    time_train_mean_comb.append(fit_mean)
    time_test_mean_comb.append(score_mean)
    file_name = "times_" + data_set + "_ada.png"
    my_plot(training_sizes, fit_mean, training_sizes, score_mean, title="AdaBoosting Learning Curve", x_label="% of Training Size", y_label="Time", fname=file_name, data_set=data_set)

    #VALIDATIONS
    param_range = np.arange(20,250,5)
    train_scores, test_scores = validation_curve(AdaBoostClassifier(n_estimators=25), X_train, y_train, param_name="n_estimators", param_range=param_range, cv=10, scoring="accuracy", n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_train_mean_comb.append(train_mean)
    val_test_mean_comb.append(test_mean)
    file_name = "valid_esti" + data_set + "_ada.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="AdaBoosting Validation Curve With AdaBoostClassifier", x_label="Number of Estimators", y_label="Accuracy", fname=file_name, data_set=data_set)
    param_range = np.arange(1, 4, 2)
    param_range = np.linspace(0.1, 2.0, 20)
    param_range = [0.1, 0.5, 0.75, 1.0]
    train_scores, test_scores = validation_curve(AdaBoostClassifier(), X_train, y_train, param_name="learning_rate", param_range=param_range, cv=10, scoring="accuracy", n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_train_mean_comb2.append(train_mean)
    val_test_mean_comb2.append(test_mean)
    file_name = "cv_" + data_set + "_ada.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="AdaBoosting Cross-Validation score", x_label="Learning Rate", y_label="Accuracy", fname=file_name, data_set=data_set)

    #PRINT ACCURACY
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:", data_set,"AdaBoosing", metrics.accuracy_score(y_test, y_pred))

########### run with best parameters
    clf = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = 0.5, n_estimators= 100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter - Accuracy:", data_set,"AdaBoosing", metrics.accuracy_score(y_test, y_pred))

    clf = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = 0.5, n_estimators= 80)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter - Accuracy:", data_set,"AdaBoosing", metrics.accuracy_score(y_test, y_pred))

def experiment_svm(X_train, X_test, y_train, y_test, data_set="data_set"):

    training_sizes=np.linspace(0.1, 1.0, 10)
    #ACCURACY and TIMES
    start_time = time.time()
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(SVC(kernel="rbf"), X=X_train, y=y_train, scoring='accuracy', cv=10, n_jobs=-1, return_times=True, train_sizes=training_sizes)
    end_time = time.time()
    print("SVM Training time=", end_time - start_time)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    acc_train_mean_comb.append(train_mean)
    acc_test_mean_comb.append(test_mean)
    acc_test_mean_comb_all.append(test_mean)
    fit_train_mean = np.mean(fit_times, axis=1)
    fit_train_times_comb_all.append(fit_train_mean)
    file_name = "acc_" + data_set + "_svm.png"
    my_plot(training_sizes, train_mean, training_sizes, test_mean, title="SVM(rbf) Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname=file_name, data_set=data_set)
    fit_mean = np.mean(fit_times, axis=1)
    score_mean = np.mean(score_times, axis=1)
    fit_mean_comb.append(fit_mean)
    score_mean_comb.append(score_mean)
    time_train_mean_comb.append(fit_mean)
    time_test_mean_comb.append(score_mean)
    file_name = "times_" + data_set + "_svm.png"
    my_plot(training_sizes, fit_mean, training_sizes, score_mean, title="SVM(rbf) Times ", x_label="% of Training Size", y_label="Time", fname=file_name, data_set=data_set)

    start_time = time.time()
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(SVC(kernel="linear"), X=X_train, y=y_train, scoring='accuracy', cv=10, n_jobs=-1, return_times=True, train_sizes=training_sizes)
    end_time = time.time()
    print("SVM Linear Kernel Training time=", end_time - start_time)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    file_name = "acc_lin_" + data_set + "_svm.png"
    my_plot(training_sizes, train_mean, training_sizes, test_mean, title="SVM(linear) Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname=file_name, data_set=data_set)
    fit_mean = np.mean(fit_times, axis=1)
    score_mean = np.mean(score_times, axis=1)
    file_name = "times_lin_" + data_set + "_svm.png"
    my_plot(training_sizes, fit_mean, training_sizes, score_mean, title="SVM(linear) Times ", x_label="% of Training Size", y_label="Time", fname=file_name, data_set=data_set)

    #VALIDATIONS
    param_range = np.arange(1000, 10000, 200)
    train_scores, test_scores = validation_curve(SVC(kernel="linear"), X_train, y_train, param_name="max_iter", param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_train_mean_comb2.append(train_mean)
    val_test_mean_comb2.append(test_mean)
    file_name = "val_max_iter_" + data_set + "_svm.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="SVM Validation Curve With max_iter", x_label="max_iter", y_label="Accuracy", fname=file_name, data_set=data_set)

    param_range = np.arange(2, 10, 2)
    train_scores, test_scores = validation_curve(SVC(kernel="linear"), X_train, y_train, param_name="C", param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    file_name = "val_penalty_" + data_set + "_svm.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="SVM Validation Curve With plot_penalty_parameter_validation_curve/C", x_label="x_label", y_label="Accuracy", fname=file_name, data_set=data_set)

    param_range=['linear', 'poly', 'rbf', 'sigmoid']
    train_scores, test_scores = validation_curve(SVC(kernel="linear"), X_train, y_train, param_name="kernel", param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_train_mean_comb.append(train_mean)
    val_test_mean_comb.append(test_mean)
    file_name = "val_kernel" + data_set + "_svm.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="SVM Validation Curve With kernels", x_label="x_label", y_label="Accuracy", fname=file_name, data_set=data_set)
    #PRINT ACCURACY
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:", data_set,"Support Vector Machine", metrics.accuracy_score(y_test, y_pred))

########### run with best parameters
    clf = SVC(C= 0.75, degree= 2, kernel= 'rbf', max_iter= 200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter - Accuracy:", data_set,"Support Vector Machine", metrics.accuracy_score(y_test, y_pred))

    clf = SVC(C= 0.75, degree= 2, kernel= 'rbf', max_iter= 200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter - Accuracy:", data_set,"Support Vector Machine", metrics.accuracy_score(y_test, y_pred))

def experiment_knn(X_train, X_test, y_train, y_test, data_set="data_set"):

    training_sizes=np.linspace(0.1, 1.0, 10)
    start_time = time.time()
    train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(KNeighborsClassifier(), X=X_train, y=y_train, scoring='accuracy', cv=10, n_jobs=-1, return_times=True, train_sizes=training_sizes)
    end_time = time.time()
    print("KNN Training time=", end_time - start_time)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    file_name = "acc_" + data_set + "_knn.png"
    my_plot(training_sizes, train_mean, training_sizes, test_mean, title="KNN Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname=file_name, data_set=data_set)
    acc_train_mean_comb.append(train_mean)
    acc_test_mean_comb.append(test_mean)
    acc_test_mean_comb_all.append(test_mean)
    fit_train_mean = np.mean(fit_times, axis=1)
    fit_train_times_comb_all.append(fit_train_mean)
    print("acc_test_mean_comb_all=", acc_test_mean_comb_all)

    fit_mean = np.mean(fit_times, axis=1)
    score_mean = np.mean(score_times, axis=1)
    fit_mean_comb.append(fit_mean)
    score_mean_comb.append(score_mean)
    time_train_mean_comb.append(fit_mean)
    time_test_mean_comb.append(score_mean)
    file_name = "times_" + data_set + "_knn.png"
    my_plot(training_sizes, fit_mean, training_sizes, score_mean, title="KNN Times", x_label="% of Training Size", y_label="Time", fname=file_name, data_set=data_set)

    param_range = np.arange(5, 50, 2)

    train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, param_name="n_neighbors", param_range=param_range, cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_train_mean_comb.append(train_mean)
    val_test_mean_comb.append(test_mean)
    file_name = "val" + data_set + "_knn.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="KNN Validation Curve With K Nearest Neighbors", x_label="Numbers of Neighbors", y_label="Accuracy", fname=file_name, data_set=data_set)

#    param_range = ['uniform', 'distance']
#    train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, param_name="weights", param_range = param_range, cv=10, scoring="accuracy", n_jobs=-1)
#    train_mean = np.mean(train_scores, axis=1)
#    test_mean = np.mean(test_scores, axis=1)
#    val_train_mean_comb2.append(train_mean)
#    val_test_mean_comb2.append(test_mean)
#    param_range = ["uniform", "distance"]
#    file_name = "val2" + data_set + "_knn.png"
#    my_plot(param_range, train_mean, param_range, test_mean, title="KNN Cross-Validation score with Weights", x_label="Weights", y_label="Accuracy", fname=file_name, data_set=data_set)
    param_range = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
    train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, param_name="metric", param_range = param_range, cv=10, scoring="accuracy", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    val_train_mean_comb2.append(train_mean)
    val_test_mean_comb2.append(test_mean)
    file_name = "val2" + data_set + "_knn.png"
    my_plot(param_range, train_mean, param_range, test_mean, title="KNN Cross-Validation score with Weights", x_label="Weights", y_label="Accuracy", fname=file_name, data_set=data_set)

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    x_test_acc = []
    y_test_acc = []
    plt.close()
    #max_limit = len(y_train)
    for n in range(1,50,2):
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_test_pred)
        #print("Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
        x_test_acc.append(n)
        y_test_acc.append(acc)
        #print("y_test_pred=")
        #print_df(y_test_pred)
    print("test accuracy=",x_test_acc, y_test_acc)

    x_train_acc = []
    y_train_acc = []
    for n in range(1,50,2):
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        acc = metrics.accuracy_score(y_train, y_train_pred)
        #print("Accuracy:",metrics.accuracy_score(y_train, y_train_pred))
        x_train_acc.append(n)

        y_train_acc.append(acc)
        #print("y_train_pred=")
        #print_df(y_train_pred)
    file_name = "acc_nn_counts_" + data_set + "_knn.png"
    my_plot(x_train_acc, y_train_acc, x_test_acc, y_test_acc, title="KNN Nearest Neighbor vs Accuracy", x_label="#Nearest Neighbors", y_label="Accuracy", fname=file_name, data_set=data_set)

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Default Parameter -Accuracy:",metrics.accuracy_score(y_test, y_pred))

    clf = KNeighborsClassifier(algorithm= 'auto', leaf_size= 1, n_neighbors= 7, p= 2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter KNeighborsClassifier - Accuracy:",metrics.accuracy_score(y_test, y_pred))

    clf = KNeighborsClassifier(algorithm= 'auto', leaf_size= 1, n_neighbors= 7, p= 2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("With Best parameter KNeighborsClassifier - Accuracy:",metrics.accuracy_score(y_test, y_pred))

##### MAIN 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt',action='store_true')
    parser.add_argument('-ada',action='store_true')
    parser.add_argument('-ann',action='store_true')
    parser.add_argument('-svm',action='store_true')
    parser.add_argument('-knn',action='store_true')
    parser.add_argument('-stat',action='store_true')
    parser.add_argument('-bp',action='store_true')
    parser.add_argument('-all',action='store_true')
    args = parser.parse_args()

##### RUN 
    #MAIN
    rng = np.random.RandomState(2)
    #X1, y1 = make_classification( n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    X1, y1 = make_classification( n_samples=2500, n_features=10, n_redundant=0, n_informative=10, random_state=rng, flip_y=0.01, n_clusters_per_class=1)
##    X1, y1 = make_classification( n_samples=1500, n_features=10, n_redundant=0, n_informative=10, random_state=rng, flip_y=0.01, n_clusters_per_class=1)
    #X1, y1 = make_classification( n_samples=500, n_features=20, n_redundant=0, n_informative=20, random_state=rng, flip_y=0.01, n_clusters_per_class=1)
    #X1, y1 = make_classification( n_samples=520, n_features=10, n_redundant=0, n_informative=10, random_state=rng, flip_y=0.01, n_clusters_per_class=1)
    X1 = StandardScaler().fit_transform(X1)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=.20, stratify=y1)
    print("Data size:", X1.shape[0], "X_train_size:", len(X_train1), "X_test_size:", len(X_test1))


    #X, y = make_classification( n_samples=50000, n_features=30, n_redundant=0, n_informative=30, random_state=rng, flip_y=0.08, n_classes=4, n_clusters_per_class=1, weights=[0.2, 0.3, 0.15,0.35])
    ##X, y = make_classification( n_samples=50000, n_features=30, n_redundant=0, n_informative=30, random_state=rng, flip_y=0.01, n_classes=2, n_clusters_per_class=1, weights=[0.4, 0.5])
    ###X, y = make_classification( n_samples=5000, n_features=30, n_redundant=0, n_informative=30, random_state=rng, flip_y=0.01, n_classes=4, n_clusters_per_class=1)

    #X, y = make_classification( n_samples=5000, n_features=20, n_redundant=0, n_informative=20, random_state=rng, n_classes=4, flip_y=0.1, n_clusters_per_class=1)
    X, y = make_classification( n_samples=15000, n_features=30, n_redundant=2, n_informative=25, random_state=rng, n_classes=4, flip_y=0.1, n_clusters_per_class=1)
##    X, y = make_classification( n_samples=5000, n_features=30, n_redundant=2, n_informative=25, random_state=rng, n_classes=4, flip_y=0.1, n_clusters_per_class=1)

    #X, y = make_classification( n_samples=500, n_features=30, n_redundant=0, n_informative=30, random_state=rng, flip_y=0.08, n_classes=2, n_clusters_per_class=1)
    #X, y = make_classification( n_samples=520, n_features=10, n_redundant=0, n_informative=10, random_state=rng, flip_y=0.10, n_classes=4, n_clusters_per_class=1)
    X = StandardScaler().fit_transform(X)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=.20, random_state=rng, stratify=y)
    print("Data size:", X.shape[0], "X_train_size:", len(X_train2), "X_test_size:", len(X_test2))

    #find_best_params(X_train1, X_test1, y_train1, y_test1, data_set="dataset1", la="ann")
    #find_best_params(X_train2, X_test2, y_train2, y_test2, data_set="dataset2", la="ann")
    #exit(0)
    #test_default(X_train1, X_test1, y_train1, y_test1, data_set="dataset1")
    #test_default(X_train2, X_test2, y_train2, y_test2, data_set="dataset2")
    #exit(0)
#
    ####if args.bp or args.all:
    if args.bp:
        print("Experiment with the data_set 1")
        find_best_params(X_train1, X_test1, y_train1, y_test1, data_set="dataset1", la="dt")
        find_best_params(X_train1, X_test1, y_train1, y_test1, data_set="dataset1", la="knn")
        find_best_params(X_train1, X_test1, y_train1, y_test1, data_set="dataset1", la="svm")
        find_best_params(X_train1, X_test1, y_train1, y_test1, data_set="dataset1", la="ada")
        find_best_params(X_train1, X_test1, y_train1, y_test1, data_set="dataset1", la="nn")
        #exit(0)
        print("Experiment with the data_set 2")
        find_best_params(X_train2, X_test2, y_train2, y_test2, data_set="dataset2", la="dt")
        find_best_params(X_train2, X_test2, y_train2, y_test2, data_set="dataset2", la="knn")
        find_best_params(X_train2, X_test2, y_train2, y_test2, data_set="dataset2", la="svm")
        find_best_params(X_train2, X_test2, y_train2, y_test2, data_set="dataset2", la="ada")
        find_best_params(X_train2, X_test2, y_train2, y_test2, data_set="dataset2", la="nn")


    if args.dt or args.all:
        experiment_dt(X_train1, X_test1, y_train1, y_test1, data_set="dataset1")
        #exit(1)
        experiment_dt(X_train2, X_test2, y_train2, y_test2, data_set="dataset2")
        my_plot_comb(training_sizes, acc_train_mean_comb,  acc_test_mean_comb, title="Decision Tree Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname="acc_comb_dt.png")
        my_plot_comb(training_sizes, time_train_mean_comb,  time_test_mean_comb, title="Decision Tree Times Curve(Ds1 and DS1)", x_label="% of Training Size", y_label="Time", fname="time_comb_dt.png")
        #my_plot_comb(training_sizes, fit_mean_comb,  score_mean_comb, title="Decision Tree Validation Curve Times", x_label="% of Training Size", y_label="Time(Sec)", fname="times_comb_dt.png")
        param_range = np.arange(1, 50, 2)
        my_plot_comb(param_range, val_train_mean_comb,  val_test_mean_comb, title="Decision Tree Validation Curve(Depth of Tree)", x_label="Depth of Tree", y_label="Accuracy", fname="val_comb_dt.png")
        my_plot_comb(param_range, val_train_mean_comb2,  val_test_mean_comb2, title="Decision Tree Validation Curve Leaf Count", x_label="Leaf Count", y_label="Accuracy", fname="val2_comb_dt.png")
        fit_mean_comb.clear()
        score_mean_comb.clear()
        acc_train_mean_comb.clear()
        acc_test_mean_comb.clear()
        time_train_mean_comb.clear()
        time_test_mean_comb.clear()
        val_train_mean_comb.clear()
        val_test_mean_comb.clear()
        val_train_mean_comb2.clear()
        val_test_mean_comb2.clear()
        #exit(0)

    if args.ann or args.all:
        experiment_ann(X_train1, X_test1, y_train1, y_test1, data_set="dataset1")
        experiment_ann(X_train2, X_test2, y_train2, y_test2, data_set="dataset2")
        print("NN: ", training_sizes, acc_train_mean_comb,  acc_test_mean_comb)
        my_plot_comb(training_sizes, acc_train_mean_comb,  acc_test_mean_comb, title="Neural Network Learning Curve", x_label="% of % of Training Size", y_label="Accuracy", fname="acc_comb_ann.png")
        my_plot_comb(training_sizes, time_train_mean_comb,  time_test_mean_comb, title="Neural Networks Times Curve(Ds1 and DS1)", x_label="% of Training Size", y_label="Time", fname="time_comb_ann.png")
        #param_range = np.arange(20,70,5)
        param_range = np.arange(200, 1000, 200)
        param_range = ["identity", "logistic", "tanh", "relu"]
        my_plot_comb(param_range, val_train_mean_comb,  val_test_mean_comb, title="Neural Network Validation Curve(Activation function)", x_label="Activation function", y_label="Accuracy", fname="val_comb_ann.png")
        #param_range = np.array([10, 15, 20, 25, 30,40, 50, 100])
        param_range = np.array([10, 20, 25, 30, 50, 100,150])
        my_plot_comb(param_range, val_train_mean_comb2,  val_test_mean_comb2, title="Neural Network Validation Curve (Numbers of Neurons)", x_label="Numbers of Neurons", y_label="Accuracy", fname="val2_comb_ann.png")
        acc_train_mean_comb.clear()
        acc_test_mean_comb.clear()
        time_train_mean_comb.clear()
        time_test_mean_comb.clear()
        val_train_mean_comb.clear()
        val_test_mean_comb.clear()
        val_train_mean_comb2.clear()
        val_test_mean_comb2.clear()

    if args.ada or args.all:
        experiment_ada(X_train1, X_test1, y_train1, y_test1, data_set="dataset1")
        experiment_ada(X_train2, X_test2, y_train2, y_test2, data_set="dataset2")
        my_plot_comb(training_sizes, acc_train_mean_comb,  acc_test_mean_comb, title="AdaBoosting Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname="acc_comb_ada.png")
        #param_range = np.arange(20,100,5)
        param_range = np.arange(20,250,5)
        my_plot_comb(param_range, val_train_mean_comb,  val_test_mean_comb, title="AdaBoosting Validation Curve(Number of Estimators)", x_label="Number of Estimators", y_label="Accuracy", fname="val_comb_ada.png")
        #param_range = np.arange(1, 4, 2)
        param_range = np.linspace(0.1, 2.0, 20)
        param_range = [0.1, 0.5, 0.75, 1.0]
        my_plot_comb(param_range, val_train_mean_comb2,  val_test_mean_comb2, title="AdaBoosting Validation Curve(Learning Rate)", x_label="Learning Rate", y_label="Accuracy", fname="val2_comb_ada.png")
        my_plot_comb(training_sizes, time_train_mean_comb,  time_test_mean_comb, title="AdaBoost Times Curve(Ds1 and DS1)", x_label="% of Training Size", y_label="Time", fname="time_comb_ada.png")
        acc_train_mean_comb.clear()
        acc_test_mean_comb.clear()
        time_train_mean_comb.clear()
        time_test_mean_comb.clear()
        val_train_mean_comb.clear()
        val_test_mean_comb.clear()
        val_train_mean_comb2.clear()
        val_test_mean_comb2.clear()
        ####experiment_ada(X_train, X_test, y_train, y_test)
        #experiment_ada()

    if args.svm or args.all:
        experiment_svm(X_train1, X_test1, y_train1, y_test1, data_set="dataset1")
        experiment_svm(X_train2, X_test2, y_train2, y_test2, data_set="dataset2")
        my_plot_comb(training_sizes, acc_train_mean_comb,  acc_test_mean_comb, title="SVM Classifier Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname="acc_comb_svm.png")
        my_plot_comb(training_sizes, time_train_mean_comb,  time_test_mean_comb, title="SVM Classifier Times Curve(Ds1 and DS1)", x_label="% of Training Size", y_label="Time", fname="time_comb_svm.png")
        #param_range = np.arange(1000, 2000, 200)
        param_range = ['linear', 'poly', 'rbf', 'sigmoid']
        my_plot_comb(param_range, val_train_mean_comb,  val_test_mean_comb, title="SVM  ClassifierValidation Curve (Kernels)", x_label="Kernel", y_label="Accuracy", fname="val_comb_svm.png")
        #param_range = np.arange(1000, 2000, 200)
        param_range = np.arange(1000, 10000, 200)
        my_plot_comb(param_range, val_train_mean_comb2,  val_test_mean_comb2, title="SVM Classifier Validation Curve (Maximum Iterations)", x_label=" max_iter", y_label="Accuracy", fname="val2_comb_svm.png")
        my_plot_comb(training_sizes, time_train_mean_comb,  time_test_mean_comb, title="Decision Tree Times Curve(Ds1 and DS1)", x_label="% of Training Size", y_label="Time", fname="time_comb_svm.png")
        acc_train_mean_comb.clear()
        acc_test_mean_comb.clear()
        time_train_mean_comb.clear()
        time_test_mean_comb.clear()
        val_train_mean_comb.clear()
        val_test_mean_comb.clear()
        val_train_mean_comb2.clear()
        val_test_mean_comb2.clear()
        #experiment_svm()
        ####experiment_svm(X_train, X_test, y_train, y_test)

    if args.knn or args.all:
        experiment_knn(X_train1, X_test1, y_train1, y_test1, data_set="dataset1")
        experiment_knn(X_train2, X_test2, y_train2, y_test2, data_set="dataset2")
        my_plot_comb(training_sizes, acc_train_mean_comb, acc_test_mean_comb, title="KNN Learning Curve", x_label="% of Training Size", y_label="Accuracy", fname="acc_comb_knn.png")
        param_range = np.arange(5, 50, 2)
        my_plot_comb(param_range, val_train_mean_comb,  val_test_mean_comb, title="KNN Validation Curve (Number of Nearest Neighbors)", x_label="Numbers of Neighbors", y_label="Accuracy", fname="val_comb_knn.png")
#        param_range = ['uniform', 'distance']
        param_range = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
        my_plot_comb(param_range, val_train_mean_comb2,  val_test_mean_comb2, title="KNN Validation Curve (distance metric)", x_label="distance metric", y_label="Accuracy", fname="val2_comb_knn.png")
        my_plot_comb(training_sizes, time_train_mean_comb,  time_test_mean_comb, title="KNN Times Curve(Ds1 and DS1)", x_label="% of Training Size", y_label="Time", fname="time_comb_knn.png")
        acc_train_mean_comb.clear()
        acc_test_mean_comb.clear()
        time_train_mean_comb.clear()
        time_test_mean_comb.clear()
        val_train_mean_comb.clear()
        val_test_mean_comb.clear()
        val_train_mean_comb2.clear()
        val_test_mean_comb2.clear()
        #experiment_knn(X_train, X_test, y_train, y_test)

    
    #####if args.stat or args.all:
    if args.stat:
        #calc_accuracy_all(X_train1, X_test1, y_train1, y_test1, data_set="dataset1")
        #calc_accuracy_all(X_train2, X_test2, y_train2, y_test2, data_set="dataset2")
        calc_accuracy_with_bp(X_train1, X_test1, y_train1, y_test1, data_set="dataset1")
        calc_accuracy_with_bp(X_train2, X_test2, y_train2, y_test2, data_set="dataset2")


    #my_barplot(ds1, data_set="dataset1")
    #my_barplot(ds2, data_set="dataset2")
    my_plot_comb_all(training_sizes, acc_test_mean_comb_all, title="All Validation TEST Learning Curves", x_label="% of Training Size", y_label="Accuracy", fname="acc_comb_all.png")
    my_plot_comb_all(training_sizes, fit_train_times_comb_all, title="All Validation Train Times Curves", x_label="% of Training Size", y_label="Train Times", fname="times_comb_all.png")
    exit(0)
