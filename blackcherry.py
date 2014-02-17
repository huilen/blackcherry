import os
import re
import pickle
import postgresql
import math
import unicodedata
import itertools
from collections import defaultdict
from functools import reduce


HAM, SPAM = range(0, 2)
TESTING, TRAINING = range(0, 2)


class Document:

    """ A document consists in a list of terms and an optional label. """

    def __init__(self, terms, label=None):
        self.terms = terms
        self.label = label

    def __iter__(self):
        return self.terms.__iter__()


class Scoring:

    """ Keeps the document counts and frequencies of a collection of documents.

    Creating a scoring for a collection of documents:

        >>> documents = [ \
                Document(terms=['a', 'b', 'c'], label=HAM), \
                Document(terms=['a', 'f', 'e'], label=HAM), \
                Document(terms=['h', 'd', 'h'], label=HAM), \
                Document(terms=['b', 'a', 'g'], label=SPAM), \
                Document(terms=['a', 'a', 'f'], label=SPAM), \
                Document(terms=['j', 'd', 'h'], label=SPAM)]
        >>> scoring = Scoring(documents)

    To get the document counts:

        >>> scoring.dc('a', SPAM)
        2
        >>> scoring.dc(label=SPAM)
        3
        >>> scoring.dc('a')
        4
        >>> scoring.dc()
        6

    To get the document frequencies:

        >>> "%.2f" % scoring.df('a', SPAM) # dc('a', SPAM) / dc(label=SPAM)
        '0.67'
        >>> "%.2f" % scoring.df(label=SPAM) # dc(label=SPAM) / dc()
        '0.50'
        >>> "%.2f" % scoring.df('a') # dc('a') / dc()
        '0.67'

    """

    def __init__(self, documents):
        self._dc = defaultdict(lambda: defaultdict(int))
        self._df = defaultdict(lambda: defaultdict(int))
        for document in documents:
            self._dc[None][document.label] += 1
            self._dc[None][None] += 1
            for term in set(document):
                self._dc[term][document.label] += 1
                self._dc[term][None] += 1
        for term in self.terms():
            for label in self.labels():
                total = self._dc[None][label if term else None]
                self._df[term][label] = self._dc[term][label] / total

    def df(self, term=None, label=None):
        return self._df[term][label]

    def dc(self, term=None, label=None):
        return self._dc[term][label]

    def terms(self):
        return self._dc

    def labels(self):
        return self._dc[None]


class Model:

    """ The model receives a collection of documents and builds a score of them
    to make predictions of the label of the documents in function of their
    terms.

        >>> documents = [ \
                Document(terms=['a', 'b', 'c'], label=HAM), \
                Document(terms=['a', 'f', 'e'], label=HAM), \
                Document(terms=['h', 'd', 'h'], label=HAM), \
                Document(terms=['b', 'a', 'g'], label=SPAM), \
                Document(terms=['a', 'a', 'f'], label=SPAM), \
                Document(terms=['j', 'd', 'h'], label=SPAM)]
        >>> model = Model(documents)

        >>> "%.2f" % model.p(Document(terms=['a', 'd']), SPAM) # df('a', SPAM) * df('d', SPAM) * df(SPAM)
        '0.11'
        >>> "%.2f" % model.p(Document(terms=['a', 'd']), HAM) # df('a', HAM) * df('d', HAM) * df(HAM)
        '0.11'
        >>> "%.2f" % model.p(Document(terms=['a', 'e']), SPAM) # df('a', SPAM) * df('e', SPAM) * df(SPAM)
        '0.00'
        >>> "%.2f" % model.p(Document(terms=['a', 'e']), HAM) # df('a', HAM) * df('e', HAM) * df(HAM)
        '0.11'
        >>> model.classify(Document(terms=['a', 'e'])) # max(p(document, HAM), p(document, SPAM))
        0

    """

    def __init__(self, documents, file=None):
        if file:
            self._scoring = pickle.load(open(file, 'rb'))
        else:
            self._scoring = Scoring(documents)

    def save(self, file):
        pickle.dump(file, open(file, 'wb'))

    def classify(self, document):
        return max([label for label in self._scoring.labels()],
                key=lambda label: self._p(document, label))

    def _p(self, document, label):
        terms = set(self._scoring.terms()).intersection(document)
        return reduce(lambda t1, t2: t1 * self._scoring.df(t2, label), terms, 1) * self._scoring.df(None, label)


class Classifier:

    """ The classifier does the following things:

        1) Gets the training data from a repository.

        2) Builds the documents from that obtained data processing it with a tokenizer.

        3) Creates a model and trains it with the builded documents.

        4) Uses the model to classify new documents.

    """

    def __init__(self):
        self._tokenizer = Tokenizer()
        self._repository = Repository('repository.db')
        self._model = Model(self._documents(TRAINING, 5000))

    def save(self, file):
        self._model.save(file)

    def classify(self, text):
        document = Document(self._tokenizer.tokens(text))
        return self._model.classify(self, document)

    def test(self):
        total = defaultdict(int)
        corrects = defaultdict(int)
        documents = self._documents(TESTING, 1000)

        mod = len(documents) / 100 * 10
        for i, document in enumerate(documents):
            if document.label == self._model.classify(document):
                corrects[document.label] += 1
            total[document.label] += 1
            if not i % mod:
                percentage = i / len(documents) * 100
                print("%d%% label(%s) documents tested" % (percentage, document.label))

        print("%d/%d spam detected" % (corrects[SPAM], total[SPAM]))
        print("%d/%d ham detected" % (corrects[HAM], total[HAM]))
        print("%.2f%% correct" % (sum(corrects.values()) / sum(total.values()) * 100))

    def dump_terms(self, terms):
        pass

    def dump_documents(self, documents):
        prev_row = None
        duplicated = 0
        order_by = lambda document: max(self._model._p(document, SPAM), self._model._p(document, HAM))
        print("p(S)", "p(H)", sep='\t', end='')
        for i, document in enumerate(sorted(documents, key=order_by)):
            row = (self._model._p(document, SPAM), self._model._p(document, HAM))
            if row == prev_row and i < len(documents) - 1:
                duplicated += 1
            else:
                if duplicated:
                    print(" (...) %d times" % duplicated)
                else:
                    print()
                duplicated = 0
                print(*row, sep='\t', end='')
                prev_row = row
        print("\n%d total" % len(documents))

    def _documents(self, selector, limit):
        data = self._repository.get(selector, limit)
        return [Document(self._tokenizer.tokens(row[0]), label=row[1]) for row in data]


class Tokenizer:

    """ Divide the given string into a list of substrings

    Creating a tokenizer:

        >>> tokenizer = Tokenizer()
        >>> tokenizer.tokens('Con la grande polvareda, perdieron a Don Beltrán. \
                Nunca lo echaron de menos, hasta los muertos pasar.')
        ['con', 'la', 'grande', 'polvareda', 'perdieron', 'a', 'don', 'beltran',
                'nunca', 'lo', 'echaron', 'de', 'menos', 'hasta', 'los',
                'muertos', 'pasar']

    Enabling meta tokens:

        >>> tokenizer.set_meta_tokens()
        >>> tokenizer.set_size_meta_tokens()
        >>> tokenizer.tokens('Buy VIAGRA! dontbeleveus@itsallalie.com')
        ['buy', 'viagra', 'dontbeleveus', 'itsallalie', 'com',
                '__EMAIL__', '__ALLCAPS__', '__SIZE10-49__']

    """

    def __init__(self):
        self._token_pattern = re.compile("\w+")
        self._meta_tokens = None
        self._size_meta_tokens = None
        self._stop_words = None
        self._stemmer = None

    def set_meta_tokens(self, meta_tokens=None):
        if meta_tokens:
            self._meta_tokens = meta_tokens
        else:
            self._meta_tokens = {
                '__EMAIL__': re.compile('@'),
                '__URL__': re.compile('http://|www\.'),
                '__ALLCAPS__': re.compile('[A-Z]{4}')
            }

    def set_size_meta_tokens(self, size_meta_tokens=[(1, 9), (10, 49), (50, 200)]):
        self._size_meta_tokens = size_meta_tokens

    def set_stop_words(self, stop_words=None, lang='english'):
        if stop_words:
            self._stop_words = stop_words
        else:
            import nltk.corpus
            self._stop_words = nltk.corpus.stopwords.words(lang)

    def set_stemmer(self, stemmer=None, lang='english'):
        if stemmer:
            self._stemmer = stemmer
        else:
            from nltk import SnowballStemmer
            self._stemmer = SnowballStemmer(lang)

    def tokens(self, text):
        tokens = []

        for token in re.findall(self._token_pattern, text):

            if self._stemmer:
                try:
                    tokens.append(self._stemmer.stem(token))
                except:
                    pass
            else:
                token = ''.join([c for c in unicodedata.normalize(
                    'NFD', token) if not unicodedata.combining(c)]).lower()

                if not self._stop_words or token not in self._stop_words:
                    tokens.append(token)

        if self._meta_tokens:
            for meta_token, pattern in self._meta_tokens.items():
                for m in pattern.findall(text):
                    tokens.append(meta_token)

        if self._size_meta_tokens:
            for f, t in self._size_meta_tokens:
                size = len(text)
                if size >= f and size <= t:
                    tokens.append('__SIZE%d-%d__' % (f, t))
                    break

        return tokens


class Repository:

    """ Loads data from the repository.

    Creating a new repository:

        >>> repository = Repository()

    @selector determines a set target, so you may be sure that you are working
    always with different sets for training that for testing.

        >>> training_data = repository.get(selector=TESTING, limit=100)
        >>> testing_data = repository.get(selector=TRAINING, limit=100)
        >>> len([row for row in training_data if row in testing_data])
        0

    @limit determines the number of rows to get for each label.

        >>> data = repository.get(selector=TRAINING, limit=100)
        >>> len(data)
        200

    @labels is a list with the labels to filter, the default value is [SPAM, HAM].

        >>> data = repository.get(selector=TRAINING, limit=100, labels=[SPAM])
        >>> len(data)
        100

    Saving a repository:

        >>> repository.save('repository.db')

    Loading an existent repository:

        >>> repository = Repository('repository.db')

    """

    def __init__(self, file=None):
        if file:
            self._data = pickle.load(open(file, 'rb'))
        else:
            self._data = defaultdict(list)

    def save(self, file):
        pickle.dump(self._data, open(file, 'wb'))

    def get(self, selector, limit, labels=[SPAM, HAM]):
        key = '%s_%s_%s' % (selector, limit, labels)

        if key not in self._data:
            for label in labels:
                db = postgresql.open('cereza:moriarty@cerezadbenv1.livra.local/cereza')
                where_label = 'reason = 9' if label == SPAM else 'reason is null'
                where_selector = 'mod(c.id, 2) = %d' % selector
                sql = """select c.id, text from "user" u
                            join comment c on c.author_id = u.id
                            where text is not null and %s and %s and random() < 0.5
                            limit %d""" % (where_label, where_selector, limit)
                self._data[key] += [(row[1], label, row[0]) for row in db.query(sql)]

        return self._data[key]


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)