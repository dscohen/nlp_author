import argparse
import math
import numpy
import nltk.classify

from nltk import compat
from nltk.classify import SklearnClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier, Ridge
from sklearn.naive_bayes import MultinomialNB

import yelp.start

def train(classif, vectorizer, train_set, sparse):
    X, y = list(compat.izip(*train_set))
    X = vectorizer.fit_transform(X)
    return classif.fit(X, y)

# See http://www.nltk.org/_modules/nltk/classify/scikitlearn.html
def classify_many(classif, vectorizer, featuresets, round = True):
    X, _ = list(compat.izip(*featuresets))
    X = vectorizer.transform(X)
    results = classif.predict(X)
    results = [numpy.round_(x).clip(0) for x in results]
    return results

# See http://www.nltk.org/_modules/nltk/classify/util.html
def accuracy(results, test_set):
    correct = [l == r for ((fs, l), r) in zip(test_set, results)]
    correct = [x[0] and x[1] and x[2] for x in correct]
    if correct:
        return float(sum(correct))/len(correct)
    else:
        return 0

def dot(v1, v2):
    return sum([x*y for x, y in zip(v1, v2)])

def rmsle(results, test_set):
    W = [2, 3, 4]
    result = sum([(math.log(dot(W, l) + 1) - math.log(dot(W, r) + 1))**2
        for ((fs, l), r) in zip(test_set, results)])
    result = (result / len(results)) ** .5
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", choices=["bayes", "svm", "lsvc", "gboost", "ridge"],
        required=True, help="Machine learning algorithm")
    parser.add_argument("-s", "--submit", default=False, action='store_true',
        help="Whether to train and test on subsets or prepare a submission")
    args = parser.parse_args()

    sparse = True

    data = yelp.start.get_data()

    if args.algorithm == "bayes":
        classif = MultinomialNB()
    elif args.algorithm == "svm":
        classif = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    elif args.algorithm == "lsvc":
        classif = svm.LinearSVC()
    elif args.algorithm == "gboost":
        classif = GradientBoostingClassifier()
        sparse = False
    elif args.algorithm == "ridge":
        classif = Ridge()

    vectorizer = DictVectorizer(dtype=float, sparse=sparse)

    if args.submit:
        classif = train(classif, vectorizer, data, sparse)
        test_set = yelp.start.get_test_data()
        results = classify_many(classif, vectorizer, test_set)
        yelp.start.print_results(results)
    else:
        pct_train = .7
        num_train = int(len(data) * pct_train)
        train_set, test_set = data[:num_train], data[num_train:]

        classif = train(classif, vectorizer, train_set, sparse)
        results = classify_many(classif, vectorizer, test_set)

        print "Perfect Accuracy:", accuracy(results, test_set)
        print "Weighted RMSLE:", rmsle(results, test_set)