from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import slate.provider
import argparse



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
    data = r["data"]
    tags = r["tags"]
    idx_to_author = r["idx"]

    # http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    if args.algorithm == "bayes":
        clf = MultinomialNB().fit(X_train_tfidf, tags)

    docs_new = ['''This is our last exchange for the week, right? From this distance, our
    discussions seem to have started out on a suitably high and somber note with
    the death of John Kennedy Jr., but our ephemeral natures soon got the better of
    us. Before we could move on to the Republican tax cut, the Mexican banking
    crisis, or the fate of Taiwan, we found ourselves--to use a phrase from this
    unfortunate week--in a "graveyard spiral" of whimsy. And it is only with a
    great effort of will that I now refrain from referring you to my new favorite
    publication, Beer Frame: The Journal of Inconspicuous Consumption ,
    and its article on the superiority of the Hydrox to the Oreo, its salute to
    products with "edible spokescharacters" like the Pillsbury Doughboy, Charlie
    the Tuna, and Slim Jim, and finally its definitive history of the styptic
    pencil ("styptic," proclaims the editors, "is such an excellent word"). If you
    can top that for meaninglessness, you are welcome to try. In the meantime, it
    should be stated for the record that your own book on the history of the
    abortion wars is a marvel of substance and you should not necessarily be judged
    by the lowbrow online company you've been keeping all week.''']
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)

    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, idx_to_author[category]))