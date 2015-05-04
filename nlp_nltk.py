import argparse

from nltk.classify import SklearnClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

import nltk.classify
import yelp.start
import slate
import blog
import amazon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source",    choices=["slate", "amazon", "blog", "yelp"],
        required=True, help="Source to use for training")
    parser.add_argument("-a", "--algorithm", choices=["bayes", "svm", "lsvc", "gboost"],
        required=True, help="Machine learning algorithm")
    args = parser.parse_args()

    if args.source == "slate":
        data = slate.get_data()
    elif args.source == "yelp":
        data = yelp.start.get_data()
    elif args.source == "amazon":
        data = amazon.get_data()
    elif args.source == "blog":
        data = blog.get_data()

    if args.algorithm == "bayes":
        classif = SklearnClassifier(MultinomialNB())
    elif args.algorithm == "svm":
        classif = SklearnClassifier(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))
    elif args.algorithm == "lsvc":
        classif = SklearnClassifier(svm.LinearSVC())
    elif args.algorithm == "gboost":
        classif = SklearnClassifier(GradientBoostingClassifier(), sparse=False)

    pct_train = .7
    num_train = int(len(data) * pct_train)
    train_set, test_set = data[:num_train], data[num_train:]

    classif = classif.train(train_set)
    print "Accuracy:", nltk.classify.accuracy(classif, test_set)


