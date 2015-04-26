from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import argparse
import numpy
import slate.provider

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source",    choices=["slate", "amazon"], required=True, help="Source to use for training")
    parser.add_argument("-a", "--algorithm", choices=["bayes"],           required=True, help="Machine learning algorithm")
    args = parser.parse_args()

    # A provider should have a method get(), which returns a dict with three entries:
    # "data", a list of the texts
    # "tags", the authors of the text (in the same order)
    # "idx", a map from a tags index to an author name.

    if args.source == "slate":
        provider = slate.provider.Provider()

    r = provider.get()
    pct_train = .8
    num_train = int(len(r["data"]) * pct_train)

    train_data = r["data"][:num_train]
    test_data  = r["data"][num_train:]
    train_tags = r["tags"][:num_train]
    test_tags  = r["tags"][num_train:]

    idx_to_author = r["idx"]

    # http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    if args.algorithm == "bayes":
        clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])
        clf = clf.fit(train_data, train_tags)

    predicted = clf.predict(test_data)

    print "%s accuracy: %.2f" % (args.algorithm.title(), numpy.mean(predicted == test_tags))