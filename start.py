from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tornado import web, ioloop
from tornado.log import enable_pretty_logging

import argparse
import json
import logging
import numpy
import random
import slate.provider
import tornado
import blog.provider

def train(args):
    # A provider should have a method get(), which returns a dict with three entries:
    # "data", a list of the texts
    # "tags", the authors of the text (in the same order)
    # "idx", a map from a tags index to an author name.

    if args.source == "slate":
        provider = slate.provider.Provider()
    elif args.source == "blog":
        provider = blog.provider.Provider()

    r = provider.get()

    pct_train = .8
    num_train = int(len(r["data"]) * pct_train)

    random.seed(42)
    indices = range(len(r["data"]))
    random.shuffle(indices)

    r["data"] = [r["data"][i] for i in indices]
    r["tags"] = [r["tags"][i] for i in indices]

    train_data  = r["data"][:num_train]
    train_tags  = r["tags"][:num_train]

    test_data  = r["data"][num_train:]
    test_tags  = r["tags"][num_train:]

    names = r["idx"]

    # http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    if args.algorithm == "bayes":
        clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])
    elif args.algorithm == "svm":
        clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
        ])
    elif args.algorithm == "lsvc":
        clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', svm.LinearSVC()),
        ])

    clf = clf.fit(train_data, train_tags)
    predicted = clf.predict(test_data)

    print(metrics.classification_report(test_tags, predicted, target_names=names))
    print "%s accuracy: %.2f" % (args.algorithm.title(), numpy.mean(predicted == test_tags))

    return clf, names

class QueryHandler(web.RequestHandler):
    def initialize(self, clf, names):
        self.clf = clf
        self.names = names

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")

    def get(self):
        data = self.get_argument("snippet")

        self.write(json.dumps({"result": names[clf.predict([data])]}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source",    choices=["slate", "amazon", "blog"],
        required=True, help="Source to use for training")
    parser.add_argument("-a", "--algorithm", choices=["bayes", "svm", "lsvc"],
        required=True, help="Machine learning algorithm")
    parser.add_argument("-p", "--port", default=55555, help="Port to start tornado server on.")
    parser.add_argument("-S", "--server", default=False, action='store_true')
    args = parser.parse_args()

    clf, names = train(args)

    print names[clf.predict(["this is a test of a snippet being sent to a clf"])]

    if args.server:
        enable_pretty_logging()

        application = web.Application([
            (r"/query", QueryHandler, dict(clf = clf, names = names)),
        ])

        logging.info("Started server on port %d" % (args.port))
        application.listen(args.port)
        ioloop.IOLoop.instance().start()