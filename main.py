from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sknn.mlp import Classifier, Layer
from sklearn.metrics import confusion_matrix
import pandas as pd
import os.path
import time
from sklearn.externals import joblib

__author__ = 'timothymiko'


class DataHandler:

    def __init__(self):

        if os.path.exists('data_models/input/x_all.pkl'):
            self.X_all = joblib.load('data_models/input/x_all.pkl')
            self.Y_all = joblib.load('data_models/input/y_all.pkl')

            self.X_train = joblib.load('data_models/input/x_train.pkl')
            self.Y_train = joblib.load('data_models/input/y_train.pkl')

            self.X_test = joblib.load('data_models/input/x_test.pkl')
            self.Y_test = joblib.load('data_models/input/y_test.pkl')
        else:
            # Import the data
            df = pd.read_csv('data_models/input/train.csv')

            # Clean the data up
            df['Transmission'] = df['Transmission'].map({"AUTO": "AUTO", "MANUAL": "MANUAL", "Manual": "MANUAL"})

            for column in ['Size', 'Make', 'Color', 'Model', 'Transmission', 'WheelType']:
                df[column] = df[column].astype('category')

            # Select the most influential columns determined in analysis.py
            feature_cols = [col for col in df.columns if col in ['VehOdo', 'VehBCost', 'Color', 'WheelType', 'WarrantyCost',
                                                                 'Model', 'SubModel', 'VehicleAge']]

            df_filtered = df[feature_cols]

            # Drop rows that contain any nans
            #df_filtered = df_filtered.dropna(axis=1, how='any')

            # transform categorical attributes into numerical attributes
            df_filtered = pd.get_dummies(df_filtered)

            df_filtered_good = df_filtered[(df.IsBadBuy == 0)]
            df_filtered_bad = df_filtered[(df.IsBadBuy == 1)]

            y = df['IsBadBuy']

            # Separate into training and test data
            train_num_good_samples = 42500
            train_num_bad_samples = 7500

            self.X_all = df_filtered
            self.Y_all = y

            self.X_train = pd.concat([df_filtered_good.sample(n=train_num_good_samples),
                                      df_filtered_bad.sample(n=train_num_bad_samples)],
                                     axis=0)
            self.Y_train = y[self.X_train.index.values.tolist()]

            self.X_test = df_filtered.drop(self.X_train.index.values.tolist())
            self.Y_test = y[self.X_test.index.values.tolist()]

            joblib.dump(self.X_all, 'data_models/input/x_all.pkl')
            joblib.dump(self.Y_all, 'data_models/input/y_all.pkl')
            joblib.dump(self.X_train, 'data_models/input/x_train.pkl')
            joblib.dump(self.Y_train, 'data_models/input/y_train.pkl')
            joblib.dump(self.X_test, 'data_models/input/x_test.pkl')
            joblib.dump(self.Y_test, 'data_models/input/y_test.pkl')


class BaseEstimator:

    def __init__(self, clf, name, recreate_model=False):
        self.classifier = clf
        self.name = name
        self.model_path = 'data_models/{0}.pkl'.format(self.name)
        self.recreate_model = recreate_model

    def run(self):
        data = DataHandler()
        x_train = data.X_train
        y_train = data.Y_train
        x_test = data.X_test
        y_test = data.Y_test

        if os.path.exists(self.model_path) and not self.recreate_model:
            self.classifier = joblib.load(self.model_path)
        else:
            self.classifier.fit(x_train, y_train)
            joblib.dump(self.classifier, self.model_path)

        result = self.classifier.predict(x_test)

        print '{0} Results:'.format(self.name)
        print 'Overall Accuracy: {0:3f}%'.format(self.classifier.score(x_test, y_test) * 100)

        x_test_bad = x_test[y_test == 1]
        y_test_bad = y_test[x_test_bad.index.values.tolist()]
        print 'Bad Accuracy: {0:3f}%'.format(self.classifier.score(x_test_bad, y_test_bad) * 100)

        x_test_good = x_test[y_test == 0]
        y_test_good = y_test[x_test_good.index.values.tolist()]
        print 'Good Accuracy: {0:3f}%'.format(self.classifier.score(x_test_good, y_test_good) * 100)

        print 'Confusion matrix:'
        print confusion_matrix(result, y_test, labels=[0, 1])


start_time = time.time()

RandomForest = False
NaiveBayes = False
NeuralNetwork = False
DecisionTree = False
Adaboost = True


if RandomForest:
    BaseEstimator(
        RandomForestClassifier(),
        "Random Forest"
    ).run()

if NaiveBayes:
    BaseEstimator(
        GaussianNB(),
        "Naive Bayes"
    ).run()

if DecisionTree:
    BaseEstimator(
        DecisionTreeClassifier(class_weight={0: 1, 1: 7}),
        "Decision Tree"
    ).run()

if Adaboost:
    BaseEstimator(
        AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(class_weight={0: 7, 1: 1}),
            n_estimators=100
        ),
        "Adaboost"
    ).run()

if NeuralNetwork:
    data = DataHandler()
    x_train = data.X_train.as_matrix()
    y_train = data.Y_train.as_matrix()
    x_test = data.X_test.as_matrix()
    y_test = data.Y_test.as_matrix()

    RecreateModel = False

    if os.path.exists('data_models/neural_network.pkl') and RecreateModel:
            nn = joblib.load('data_models/neural_network.pkl')
    else:
        nn = Classifier(
            layers=[
                Layer("Sigmoid", units=10),
                Layer("Softmax")],
            learning_rate=0.9,
            n_iter=25)
        nn.fit(x_train, y_train)
        joblib.dump(nn, 'data_models/neural_network.pkl')

    result = nn.predict(x_test)

    print 'Neural Network Results:'
    print 'Overall Accuracy: {0:3f}%'.format(nn.score(x_test, y_test) * 100)

    x_test_bad = data.X_test[data.Y_test == 1]
    y_test_bad = data.Y_test[x_test_bad.index.values.tolist()]
    print 'Bad Accuracy: {0:3f}%'.format(nn.score(x_test_bad.as_matrix(), y_test_bad.as_matrix()) * 100)

    x_test_good = data.X_test[data.Y_test == 0]
    y_test_good = data.Y_test[x_test_good.index.values.tolist()]
    print 'Good Accuracy: {0:3f}%'.format(nn.score(x_test_good.as_matrix(), y_test_good.as_matrix()) * 100)

    print 'Confusion matrix:'
    print confusion_matrix(result, y_test, labels=[0, 1])


run_time = time.time() - start_time
print 'Total time elapsed is {0} milliseconds'.format(run_time)
