#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("../text_learning")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn import decomposition
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt

def create_fraction_email_features(data_dict):
    """
    funtion to create 2 new features: fraction_from_poi and fraction_to_poi
    """
    def computeFraction( poi_messages, all_messages ):
        """ given a number messages to/from POI (numerator)
            and number of all messages to/from a person (denominator),
            return the fraction of messages to/from that person
            that are from/to a POI
        """
        fraction = 0.
        if all_messages !=0 and all_messages !='NaN' and poi_messages != 'NaN':
            fraction = poi_messages/float(all_messages)
        return fraction

    for name in data_dict:

        data_point = data_dict[name]

        to_messages = data_point["to_messages"]
        fraction_from_poi = computeFraction( data_point['from_poi_to_this_person'],
                                             data_point['to_messages'] )
        data_point["fraction_from_poi"] = fraction_from_poi
        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = computeFraction( data_point['from_this_person_to_poi'],
                                           data_point['from_messages'] )
        data_point["fraction_to_poi"] = fraction_to_poi
    return data_dict

def text_learning_experiment(words_to_remove=[]):
    from_sara  = open("../text_learning/from_sara.txt", "r")
    from_chris = open("../text_learning/from_chris.txt", "r")
    word_data, authors = vectorize_emails(from_sara, from_chris, max_emails=300, words_to_remove=words_to_remove)
    features_train, features_test, labels_train, labels_test = \
        cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    features_train = vectorizer.fit_transform(features_train)
    features_test  = vectorizer.transform(features_test).toarray()

    features_train = features_train[:150].toarray()
    labels_train   = labels_train[:150]

    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    predict_train = clf.predict(features_train)
    predict_test = clf.predict(features_test)
    print "train acc:", accuracy_score(labels_train, predict_train)
    print "test acc: ", accuracy_score(labels_test, predict_test)
    feature_index = np.argmax(clf.feature_importances_)
    feature_importance = clf.feature_importances_[feature_index]
    feature_name = vectorizer.get_feature_names()[feature_index]
    print "Most important feature, and relative importance:", feature_name, ":", feature_importance
    return feature_name, feature_importance


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

################################################################################
################################################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### "poi" will be used as the label that we predict
# First investigate to decide
# Find percentage of non-NaN in each feature

feat = data_dict[data_dict.keys()[0]].keys()
feat.remove('poi')
feat.remove('email_address')
features_list = ['poi'] + feat # just place 'poi' at the beginning of the list

### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
keys, data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

###### create a function for this part #########
feature_availability = {} # percent of non-nan values in the features
for f in features_list[1:]:
    non_nan=0
    for name in keys:
        if data_dict[name][f] != 'NaN':
            non_nan += 1
    feature_availability[f] = float(non_nan)/len(keys)

print 'Features sorted by percentage of non-Nan values'
for k, v in sorted(feature_availability.items(), key = lambda x: x[1], reverse = True):
    print k, ":", v
#################################################

# I create the features_list again:

features_list = [
    'poi',
    'total_stock_value',
    'total_payments',
    'restricted_stock',
    'exercised_stock_options',
    'salary',
    'expenses',
    'other',
    'to_messages',
    'shared_receipt_with_poi',
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'bonus'
]
### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
keys, data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

################################################################################
################################################################################
### Task 2: Remove outliers
print "Removing outliers"
# Strategy:
#
# 1. PCA
# 2. plot 1st and 2nd components
# 3. visualy inspect if we have outliers
# 4. if we have outliers, remove them, repeat from step 1

# For pretty plots
isPoi = np.array(labels, dtype=bool)
isNotPoi = np.array([not p for p in isPoi])

pca = decomposition.PCA(n_components=len(features_list)-1)
pca.fit(features)
features_pca = pca.transform(features)
plt.figure()
plt.scatter(features_pca[isNotPoi,0], features_pca[isNotPoi,1], color='b')
plt.scatter(features_pca[isPoi,0], features_pca[isPoi,1], color='r')
plt.xlabel("First PC")
plt.ylabel("Second PC")
plt.title("Projection on first 2 Principal Components")
plt.savefig("outlier_pca1_pca2_scatter.png")
# The point with the lowest value on the first component looks like an outlier
print "Max of the first component:"
print keys[np.argmax(features_pca[:,0])]
print 'Sure, the "TOTAL" is not a person, it should not be in our data set.\n'

print "Min on the second component:"
print keys[np.argmin(features_pca[:,1])]
print "This is one of the employees, I'll leave it there.\n"
# Remove
my_dataset.pop("TOTAL")

print 'Remove "TOTAL" and repeat experiment.\n'
# Repeat

############################################################
#
# NOTE: This is not the process I would normally use !
#       I would normally look for parameters, outliers, etc,
#       mostly in an interactive console, and leave in the script
#       only the necesary parts for processing.
#       I reloaded the data and plotted the points again
#       only to show my reasoning.
#
##############################################################


### Extract features and labels from dataset for local testing
keys, data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

isPoi = np.array(labels, dtype=bool)
isNotPoi = np.array([not p for p in isPoi])

pca.fit(features)
features_pca = pca.transform(features)
plt.figure()
plt.scatter(features_pca[isNotPoi,0], features_pca[isNotPoi,1], color='b')
plt.scatter(features_pca[isPoi,0], features_pca[isPoi,1], color='r')
plt.xlabel("First PC")
plt.ylabel("Second PC")
plt.title("Projection on first 2 Principal Components")
plt.savefig("outlier_pca1_pca2_scatter2.png")
# I'll try to see what the lowest value on the 1st component
print "Highest value on the 1st component:"
print keys[np.argmax(features_pca[:,0])]
# Aaand, the one with the lowest value on 2nd component (y axis)?
print "Lowest value on the 2nd component:"
print keys[np.argmin(features_pca[:,1])]
# Ok, so they are all enron employees, I'll leave those in the data set
# Those are outliers that we want to pay attention to

################################################################################
################################################################################
### Task 3: Create new feature(s)
# The first idea that comes to mind is to use the emails.
# The problem is that a lot of words in this corpus are signature words,
# Or very specific to this particular case (we have names of people,
# internal code names maybe, all of those that act like markers only
# for the Enron case)
# Here's a small experiment with only two persons:
# print "\nText learning experiment"
# from vectorize_text import vectorize_emails
#
strength = 1.0
words_blacklist = []
# while strength > 0.1:
#     signature_word, strength = text_learning_experiment(from_sara, from_chris, words_blacklist)
#     words_blacklist.append(signature_word)
for _ in range(5):
    signature_word, strength = text_learning_experiment(words_blacklist)
    words_blacklist.append(signature_word)

print "--------------------"
print "top 5 most important words:"
print words_blacklist

### Engineering fraction_from_poi, fraction_to_poi
### the code to add those features is above, here I create the plots

plt.figure()
plt.scatter(data[isNotPoi,features_list.index('from_poi_to_this_person')],
        data[isNotPoi,features_list.index('from_this_person_to_poi')],
        color='b')
plt.scatter(data[isPoi,features_list.index('from_poi_to_this_person')],
        data[isPoi,features_list.index('from_this_person_to_poi')],
        color='r')
plt.xlabel('from_poi_to_this_person')
plt.ylabel('from_this_person_to_poi')
plt.title('Number of emails to and from poi')
plt.savefig("emails_poi.png")

data_dict = create_fraction_email_features(data_dict)

### Store to my_dataset for easy export below.
my_dataset = data_dict

features_list = features_list + ['fraction_from_poi', 'fraction_to_poi']
### Extract features and labels from dataset for local testing
keys, data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

plt.figure()
plt.scatter(data[isNotPoi,features_list.index('fraction_from_poi')],
        data[isNotPoi,features_list.index('fraction_to_poi')],
        color='b')
plt.scatter(data[isPoi,features_list.index('fraction_from_poi')],
        data[isPoi,features_list.index('fraction_to_poi')],
        color='r')
plt.xlabel('fraction_from_poi')
plt.ylabel('fraction_to_poi')
plt.title('Fraction of emails to and from poi')
plt.savefig("fraction_emails_poi.png")

################################################################################
################################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

clfs = [
    GaussianNB(),
    SVC(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    KNeighborsClassifier(),
    LogisticRegression()
]

for clf in clfs:
    test_classifier(clf, my_dataset, features_list)

################################################################################
################################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn import grid_search

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

svrs = [
    # (KNeighborsClassifier(), {'n_neighbors': [3,5,7],
    #                           'weights': ['distance'],
    #                           'metric': ['minkowski'],
    #                           'p': [1,2,3],
    #                           'n_jobs': [-1]
    #                           }),
    (DecisionTreeClassifier(), {'criterion':['gini', 'entropy'],
                                'min_samples_split': [20],
                                'min_samples_leaf': [1,3,5,7],
                                'class_weight': ['balanced'],
                                'random_state': [42]
                                })
]
for svr, parameters in svrs:
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(features_train, labels_train)
    clf = clf.estimator
    test_classifier(clf, my_dataset, features_list)

################################################################################
################################################################################
### Task 5': Analyze the Decision Tree selected by GridSearch, perform feature selection,
### Test the performance of created features

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

svr = DecisionTreeClassifier()
parameters = {'criterion':['gini', 'entropy'],
              'min_samples_split': [20],
              'min_samples_leaf': [1,3,5,7],
              'class_weight': ['balanced'],
              'random_state': [42]
             }
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(features_train, labels_train)
clf = clf.estimator
clf.fit(features_train, labels_train)
print "\nSelecting best features for Decision Tree"
print "Features Importances:"
features_importances = sorted(zip(features_list[1:],
                                  clf.feature_importances_), \
                                  key = lambda x:x[1], \
                                  reverse = True)
for feature, value in features_importances:
    print feature, ":", value
test_classifier(clf, my_dataset, features_list)
print "_____________________________\n"

print "Performance without 'fraction_from_poi' and 'fraction_to_poi'"
features_list.remove('fraction_from_poi')
features_list.remove('fraction_to_poi')
keys, data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(features_train, labels_train)
clf = clf.estimator
clf.fit(features_train, labels_train)
print "\nSelecting best features for Decision Tree"
print "Features Importances:"
features_importances = sorted(zip(features_list[1:],
                                  clf.feature_importances_), \
                                  key = lambda x:x[1], \
                                  reverse = True)
for feature, value in features_importances:
    print feature, ":", value
test_classifier(clf, my_dataset, features_list)
print "_____________________________\n"

# I'll add in the features fraction_from_poi and fraction_to_poi
# And perform an automated selection of the features that provide at least 1% usable information
features_list.append('fraction_from_poi')
features_list.append('fraction_to_poi')
keys, data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(features_train, labels_train)
clf = clf.estimator
clf.fit(features_train, labels_train)
features_importances = sorted(zip(features_list[1:],
                                  clf.feature_importances_), \
                                  key = lambda x:x[1], \
                                  reverse = True)
print "For final classifier, will pick the best of:"
for feature, value in features_importances:
    print feature, ":", value

new_features_list = [feature for feature, value in features_importances if value >= 0.01]
print new_features_list
keys, data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# clf = grid_search.GridSearchCV(svr, parameters)
# clf.fit(features_train, labels_train)
# clf = clf.estimator
clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
        max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        presort=False, random_state=42, splitter='best')
clf.fit(features_train, labels_train)
features_importances = sorted(zip(features_list[1:],
                                  clf.feature_importances_), \
                                  key = lambda x:x[1], \
                                  reverse = True)
print "Feature importances for final classifier"
for feature, value in features_importances:
    print feature, ":", value
test_classifier(clf, my_dataset, features_list)
################################################################################
################################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
