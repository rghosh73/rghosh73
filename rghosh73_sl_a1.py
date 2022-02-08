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



   
