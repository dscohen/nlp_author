from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import svm
from sklearn.linear_model import SGDClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source",    choices=["slate", "amazon"],
        required=True, help="Source to use for training")
    parser.add_argument("-a", "--algorithm", choices=["bayes", "svm", "lsvc"],
        required=True, help="Machine learning algorithm")
    args = parser.parse_args()

    if args.source == "slate":
        data = slate_data.get_data()
        test_data = data[1]
        train_data = data[0]

    if args.algorithm == "bayes":
        classif = SklearnClassifier(MultinomialNB()).train(train_data)
    if args.algorithm == "svm":
        classif = SklearnClassifier(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)).train(train_data)
    if args.algorithm == "lsvc":
        classif = SklearnClassifier(svm.LinearSVC()).train(train_data)
    classif.classify_many(test_data)


