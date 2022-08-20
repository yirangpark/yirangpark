# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rand

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.nedighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perception
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# 1) acquire data
submission_data = pd.read_csv('C:/Users/1004/git/kaggle/gender_submission.csv')
test_data = pd.read_csv('C:/Users/1004/git/kaggle/test.csv')
train_data = pd.read_csv('C:/Users/1004/git/kaggle/train.csv')
combine = [train_data, test_data]
combine

# 2) analyze by describing data
# 2-1) feature type
train_data.head()
train_data.tail()
print(train_data.columns.values)
train_data.dtypes
'''
Categorical = Survived, Sex, Embarked
Ordinal = Pclass
Continous = Age, Fare
Discrete = SibSp, Parch
mixed = Ticket, Cabin
contain errors or typos = Name
'''

# 2-2) check null, type
train_data.isnull().sum()
''' 
null: Cabin > Age > Embarked
'''

# 2-3) distribution features
train_data.info()
print('_'*40)
test_data.info()
train_data.describe()
train_data.describe(include=['O'])
'''
PassengerId: X
Survived: survived 38%
Pclass: 3 Pclass 50%
Age: Few elderly passengers
SibSp: Nearly 30% passenger had SibSp
Parch: Most passenger(>75%) didn't trvavel with parch
Fare: fares varied
Name: unique 891
Sex: male 65%
Ticket: 22% of duplicate values(unique = 681) 
Cabin: several passengers shared a cabin
Embarked: S port used by most passengers
'''

# 3) analyze by pivoting data
