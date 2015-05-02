import gensim
import start
import nltk
import gensim

def parse_sentences():
    data = start.get_yelp_id()
    word2vec_input = []
    i = 0
    for place in data.itervalues():
        for review in place.reviews:
            sents = nltk.tokenize.sent_tokenize(str(review.decode('utf-8')))
            for sent in sents:
                word2vec_input.append((nltk.tokenize.word_tokenize(sent)))

        if i%100 == 0:
            print "tokenizing"
    return word2vec_input

def encode_unicode(words):
    new_word = []
    for word in words:
        new_word.append(word.encode('utf-8'))
    return new_word


def train_word2vec(sentences):
    print "training NN"
    model = gensim.models.Word2Vec(sentences, min_count=20,workers=2)
    model.save('word2vecmodel')


if __name__ == "__main__":
    print parse_sentences()
