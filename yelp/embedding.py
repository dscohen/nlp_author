import gensim
import start
import nltk
import gensim

def parse_sentences():
    data = start.get_yelp_id()
    word2vec_input = []
    doc2vec = []
    i = 0
    for place in data.itervalues():
        for review in place.reviews:
            doc2vec.append(gensim.models.doc2vec.LabeledSentence(words = review, labels = [place.rest_id]))
            #sents = nltk.tokenize.sent_tokenize((review.decode('utf-8')))
            #for sent in sents:
                #word2vec_input.append(encode_unicode(nltk.tokenize.word_tokenize(sent)))


    return doc2vec
    #return word2vec_input

def encode_unicode(words):
    new_word = []
    for word in words:
        new_word.append(word.encode('utf-8'))
    return new_word


def train_word2vec(sentences):
    print "training NN"
    model = gensim.models.Doc2Vec(alpha=0.025, min_alpha=0.025,workers=2)  # use fixed learning rate
    model.build_vocab(sentences)
     
    for epoch in range(10):
        model.train(sentences)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    #model = gensim.models.Doc2Vec(sentences, min_count=20,workers=2)
    model.save('doc2vecmodel')


if __name__ == "__main__":
    train_word2vec(parse_sentences())
