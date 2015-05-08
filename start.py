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

W = [12, 14, 16]

def train(classif, vectorizer, train_set):
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
    result = [(math.log(dot(W, l) + 1) - math.log(dot(W, r) + 1))**2
        for ((fs, l), r) in zip(test_set, results)]
    return average(result) ** .5

def single_rmsle(results, test_set, index):
    result = [(math.log(W[index]*l[index]+1) - math.log(W[index]*r[index]+1))**2
        for ((fs, l), r) in zip(test_set, results)]
    return average(result) ** .5

def metrics_header():
    print " Algorithm \t PA \t 1s A \t 2s A \t 3s A \t R \t 1s R \t 2s R \t 3s R"

def metrics(name, results, test_set, header=False):
    acc = accuracy(results, test_set)
    r = rmsle(results, test_set)
    rs = [single_rmsle(results, test_set, i) for i in range(3)]
    print "%10s \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f \t %.3f" % \
        (name, acc[0], acc[1], acc[2], acc[3], r, rs[0], rs[1], rs[2])

def run(classif, datas, vectorizer, submit = False):
    results = []
    for stars in xrange(3):
        f = lambda x: [x[0],x[1][stars]]
        data = map(f,datas)
        if submit:
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
    return results.tolist(), test_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", choices=["bayes", "svm", "lsvc", "ridge","linear"],
        required=True, help="Machine learning algorithm")
    parser.add_argument("-f", "--feature", choices=["word2vec","doc2vec","tfidf"],
        required=False, help="Machine learning algorithm")
    parser.add_argument("-s", "--submit", default=False, action='store_true',
        help="Whether or not to prepare a submission (default = False)")
    parser.add_argument("-b", "--benchmark", default=False, action="store_true",
        help="Whether to benchmark all algorithms, ignoring -s and -a arguments (default = False)")
    args = parser.parse_args()

    vectorizer = DictVectorizer(dtype=float)
    datas = yelp.start.get_data(embedding = args.feature)

    algorithms = {
        "bayes": MultinomialNB(),
        "svm": SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),
        "lsvc": svm.LinearSVC(),
        "ridge": Ridge(),
        "linear": LinearRegression()
    }

    if args.benchmark:
        metrics_header()
        for name, classif in algorithms.iteritems():
            results, test_set = run(classif, datas, vectorizer)
            metrics(name, results, test_set)
    else:
        classif = algorithms[args.algorithm]
        results, test_set = run(classif, datas, vectorizer, args.submit)

        # Can only measure accuracy when testing, not submitting.
        if args.submit:
            yelp.start.print_results(results)
        else:
            metrics_header()
            metrics(args.algorithm, results, test_set)