import argparse
import math
import numpy

from nltk import compat
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier, Ridge, LinearRegression
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

def average(l):
    return float(sum(l))/len(l)

# See http://www.nltk.org/_modules/nltk/classify/util.html
def accuracy(results, test_set):
    equal = [[l[0] == r[0], l[1] == r[1], l[2] == r[2]]
        for ((fs, l), r) in zip(test_set, results)]
    correct = [x[0] and x[1] and x[2] for x in equal]
    sub_correct = [[x[i] for x in equal] for i in range(3)]
    return (average(correct), average(sub_correct[0]), average(sub_correct[1]), average(sub_correct[2]))

def dot(v1, v2):
    return sum([x*y for x, y in zip(v1, v2)])

def rmsle(results, test_set):
    W = [14, 16, 18]
    result = sum([(math.log(dot(W, l) + 1) - math.log(dot(W, r) + 1))**2
        for ((fs, l), r) in zip(test_set, results)])
    result = (result / len(results)) ** .5
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", choices=["bayes", "svm", "lsvc", "gboost", "ridge","linear"],
        required=True, help="Machine learning algorithm")
    parser.add_argument("-f", "--feature", choices=["word2vec","doc2vec","tfidf"],
        required=False, help="Machine learning algorithm")
    parser.add_argument("-s", "--submit", default=False, action='store_true',
        help="Whether or not to prepare a submission (default = False)")
    args = parser.parse_args()

    if args.algorithm == "bayes":
        classif = MultinomialNB()
    elif args.algorithm == "svm":
        classif = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    elif args.algorithm == "lsvc":
        classif = svm.LinearSVC()
    elif args.algorithm == "ridge":
        classif = Ridge()
    elif args.algorithm == "linear":
        classif = LinearRegression()

    vectorizer = DictVectorizer(dtype=float)

    datas = yelp.start.get_data(embedding = args.feature)
    results = []
    for stars in xrange(3):
        f = lambda x: [x[0],x[1][stars]]
        data = map(f,datas)
        if args.submit:
            classif = train(classif, vectorizer, data)
            test_set = yelp.start.get_test_data()
            results.append(classify_many(classif, vectorizer, test_set))
        else:
            pct_train = .8
            num_train = int(len(data) * pct_train)
            train_set, test_set = data[:num_train], datas[num_train:]
            classif = train(classif, vectorizer, train_set)
            results.append(classify_many(classif, vectorizer, test_set))
    results = numpy.asarray(results)
    results = numpy.transpose(results)
    results = results.tolist()

    # Can only measure accuracy when testing, not submitting.
    if args.submit:
        yelp.start.print_results(results)
    else:
        accuracy = accuracy(results, test_set)
        print "Perfect Accuracy: %.3f (%.3f, %.3f, %.3f)" % accuracy
        print "Weighted RMSLE:", rmsle(results, test_set)
