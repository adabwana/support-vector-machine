import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as skm
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score
import ISLP
from ISLP.svm import plot as plot_svm
from sklearn.model_selection import GridSearchCV
import pandas as pd

np.random.seed(42)
n = 2*500
p = 2
rng = np.random.default_rng(10)
x1 = rng.uniform(size = n) - 0.4
x2 = rng.uniform(size = n) - 0.4
X = np.stack([x1, x2], axis = 1)
y  = x1**2 - x2**2 > 0.1**2