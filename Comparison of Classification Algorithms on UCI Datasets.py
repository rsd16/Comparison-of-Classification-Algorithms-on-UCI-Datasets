import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings


warnings.filterwarnings('ignore')

labelencoder = LabelEncoder() # initializing an object of class LabelEncoder

# iris dataset.
#data = pd.read_csv('iris.data', header=None)
#print(data.columns)
#print(data[4].value_counts()) # data[4] is the class labels.
#data[4] = labelencoder.fit_transform(data[4])
#print(data[4].value_counts()) # data[4] is the class labels.
#data.to_csv('iris.data', index=None, header=None)

# wine dataset.
#data = pd.read_csv('wine.data', header=None)
#print(data.columns)
#print(data[0].value_counts()) # data[0] is the class labels.

# zoo dataset.
#data = pd.read_csv('zoo.data', header=None)
#print(data.columns)
#data.drop(data.columns[0], axis=1, inplace=True)
#print(data[17].value_counts()) # data[17] is the class labels.
#data.to_csv('zoo.data', index=None, header=None)

# lymphography dataset.
#data = pd.read_csv('lymphography.data', header=None)
#print(data.columns)
#print(data[0].value_counts()) # data[0] is the class labels.

# ionosphere dataset.
#data = pd.read_csv('ionosphere.data', header=None)
#print(data.columns)
#print(data[34].value_counts()) # data[34] is the class labels.
#data[34] = labelencoder.fit_transform(data[34])
#print(data[34].value_counts()) # data[34] is the class labels.
#data.to_csv('ionosphere.data', index=None, header=None)

# ecoli dataset
#data = pd.read_csv('ecoli.data', header=None)
#print(data.columns)
#print(data[8].value_counts()) # data[8] is the class labels.
#data.drop(data.columns[0], axis=1, inplace=True)
#data[8] = labelencoder.fit_transform(data[8])
#print(data[8].value_counts()) # data[8] is the class labels.
#data.to_csv('ecoli.data', index=None, header=None)

# seeds dataset
#data = pd.read_csv('seeds.txt', header=None)
#print(data.columns)
#print(data[7].value_counts()) # data[7] is the class labels.
#data.to_csv('seeds.data', index=None, header=None)

# banknote authentication dataset
#data = pd.read_csv('banknote_authentication.txt', header=None)
#print(data.columns)
#print(data[4].value_counts()) # data[4] is the class labels.
#data.to_csv('banknote_authentication.data', index=None, header=None)

# Pima Indians Diabetes Dataset
#data = pd.read_csv('Pima Indians Diabetes Dataset.txt', header=None)
#print(data.columns)
#print(data[8].value_counts()) # data[8] is the class labels.
#data.to_csv('Pima Indians Diabetes Dataset.data', index=None, header=None)

# sonar
#data = pd.read_csv('sonar.data', header=None)
#print(data.columns)
#print(data[60].value_counts()) # data[60] is the class labels.
#data[60] = labelencoder.fit_transform(data[60])
#print(data[60].value_counts())
#data.to_csv('sonar.data', index=None, header=None)

##todo: later, if necessary.
##dropna for labels, the rows with na labels
##delete samples below 5
##replace ? with -99999
##fillna with -99999
##category to number
##def preprocess_dataset(dataset_name, label_index):
##    data = pd.read_csv(f'{dataset_name}' + '.data', header=None)
##    data.dropna(subset=[label_index], inplace=True)
##    counts = data[label_index].value_counts()
##    data = data[data[label_index].isin(counts[counts > 5].index)]
##    data.replace('?', -99999, inplace=True)
##    data.fillna(-99999, inplace=True)

def read_split_data(dataset_name, label_index):
    data = pd.read_csv(f'{dataset_name}' + '.data', header=None)
    counts = data[label_index].value_counts()
    data = data[data[label_index].isin(counts[counts > 5].index)]
    x = data.drop(label_index, 1)
    y = data[label_index]
    #print(x.shape)
    #print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=True, stratify=y)
    return x_train, x_test, y_train, y_test

def describe_data(dataset_name, label_index):
    data = pd.read_csv(f'{dataset_name}' + '.data', header=None)
    print(dataset_name)
    print(data.describe())
    print(data[label_index].value_counts())

def evaluate(model_name, x_train, x_test, y_train, y_test):
    if model_name == 'Random Forest':
        model = RandomForestClassifier()
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier()
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_name == 'AdaBoost':
        model = AdaBoostClassifier()
    elif model_name == 'SVM':
        model = SVC()

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

dataset_names = {'banknote_authentication': 4, 'sonar': 60, 'iris': 4, 'Pima Indians Diabetes Dataset': 8, 'wine': 0, 'zoo': 16, 'lymphography': 0, 'ionosphere': 34, 'ecoli': 7, 'seeds': 7}
model_names = ['Random Forest', 'XGBoost', 'Decision Tree', 'AdaBoost', 'SVM']

for dataset_name, label_index in dataset_names.items():
    describe_data(dataset_name, label_index)
    #print(dataset_name)
    x_train, x_test, y_train, y_test = read_split_data(dataset_name, label_index)
    for model_name in model_names:
        accuracy = str(evaluate(model_name, x_train, x_test, y_train, y_test))
        print(f'Accuracy for dataset: "{dataset_name}" from "{model_name}" model is: {accuracy}')
