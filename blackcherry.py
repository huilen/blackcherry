import os
import sys
import re
import pickle
import postgresql
import unicodedata
import logging
import math
from collections import defaultdict
from functools import reduce


HAM, SPAM = range(0, 2)
TESTING, TRAINING = range(0, 2)
LIVRA, ENRON = range(0, 2)


class Document:

    """ A document consists in a list of terms and an optional label. """

    def __init__(self, terms, label=None, uniqid=None):
        self.terms = terms
        self.label = label
        self.uniqid = uniqid

    def __iter__(self):
        return self.terms.__iter__()


class Scoring:

    """ Keeps the document counts and frequencies of a collection of documents. """

    def __init__(self, documents):
        self._dc = defaultdict(lambda: defaultdict(int))
        self._df = defaultdict(lambda: defaultdict(int))
        for document in documents:
            self._dc[None][document.label] += 1
            self._dc[None][None] += 1
            for term in set(document):
                self._dc[term][document.label] += 1
                self._dc[term][None] += 1
        for term in self.terms().union({None}):
            for label in self.labels().union({None}):
                dc = self._dc[term][label]
                total = self._dc[None][label] #/ self._tc[label]
                self._df[term][label] = abs(math.log(1 + dc / total))
            dc = self._dc[term][None]
            self._df[term][None] = self._dc[None][None] #/ self._tc[None]
            self._df[None][None] = 1

    def df(self, term=None, label=None):
        logging.debug("df(%s, %s) = %f" % (term, label, self._dc[term][label]))
        return self._df[term][label]

    def dc(self, term=None, label=None):
        logging.debug("dc(%s, %s) = %f" % (term, label, self._dc[term][label]))
        return self._dc[term][label]

    def terms(self):
        return set([term for term in self._dc if term is not None])

    def labels(self):
        return set([label for label in self._dc[None] if label is not None])


class Model:

    """
    The model receives a collection of documents and builds a score of them
    to make predictions of the label of the documents in function of their
    terms.
    """

    def __init__(self, documents, features_size, file=None):
        if file:
            self._scoring = pickle.load(open(file, 'rb'))
        else:
            self._scoring = Scoring(documents)
            self._features = sorted([feature for feature in self._scoring.terms()],
                    key=self._d,
                    reverse=True)
            self._features = set(self._features[:features_size])

    def save(self, file):
        pickle.dump(file, open(file, 'wb'))

    def classify(self, document):
        return max([label for label in self._scoring.labels()],
                key=lambda label: self._p(document, label))

    def _p(self, document, label):
        def log(freq):
            return float('-inf') if freq == 0 else math.log(freq)

        def likelihood(feature):
            likelihood = self._scoring.df(feature, label)
            return log(likelihood if feature in document else 1 - likelihood)

        prior = log(self._scoring.df(label=label))

        p = reduce(lambda p, feature:
                p + likelihood(feature),
                self._features, prior)

        logging.debug("p(%s, %s) = %f" % (document.terms, label, p))
        return p

    def _d(self, feature):
        df_FL = self._scoring._df[feature][SPAM]
        df_F = self._scoring._df[feature][None]
        df_L = self._scoring._df[None][SPAM]
        numerator = df_FL - df_F * df_L
        denominator = df_F * (1 - df_F) * df_L * (1 - df_L)
        return abs(numerator / denominator ** 1/2)


class Tokenizer:

    """ Divide the given string into a list of substrings. """

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


class RepositorExhaustedError(Exception):
    pass


class Repository:

    def __init__(self):
        if os.path.isfile('.cache'):
            self._cache = pickle.load(open('.cache', 'rb'))
        else:
            self._cache = {}
        self._sources = {}
        self._sources[LIVRA] = self._get_from_livra
        self._sources[ENRON] = self._get_from_enron

    def get(self, selector, source, limit, labels=[SPAM, HAM], settings=[]):
        key = '%s_%s_%s_%s' % (selector, limit, labels, source)
        if key not in self._cache:
            logging.info("Obtaining from source: %s" % key)
            data = self._sources[source](selector, limit, labels, *settings)
            if len(data) != limit * len(labels):
                raise DataSetExhaustedError
            else:
                self._cache[key] = data
                pickle.dump(self._cache, open('.cache', 'wb'))
            logging.info("Obtained from source: %s" % len(self._cache[key]))
        else:
            logging.info("Obtained from cache: %s" % len(self._cache[key]))

        return self._cache[key]

    def _get_from_livra(self, selector, limit, labels, where_custom=''):
        rows = []
        for label in labels:
            db = postgresql.open('cereza:moriarty@cerezadbenv1.livra.local/cereza')
            where_label = 'reason = 9' if label == SPAM else 'reason is null'
            where_selector = 'mod(c.id, 2) = %d' % selector
            sql = """select min(c.id), text from "user" u
                        join comment c on c.author_id = u.id
                        where text is not null and %s and %s and %s
                        group by text
                        limit %d""" % (where_label, where_selector,
                                where_custom, limit)
            rows += [(row[1], label, row[0]) for row in db.query(sql)]
        return rows

    def _get_from_enron(self, selector, limit, labels):
        rows = []
        for label in labels:
            directory = 'corpus/enron/%s/%s' % (selector, label)
            for f in os.listdir(directory)[:limit]:
                with open(os.path.join(directory, f), 'rb') as f:
                    rows.append((f.read().decode('latin-1'), label, f.name))
        return rows


class Test:

    def __init__(self, name):
        self.name = name

    def run(self):
        if os.path.exists('tests/%s' % self.name):
            for f in os.listdir('tests/%s' % self.name):
                os.remove('tests/%s/%s' % (self.name, f))
        else:
            os.makedirs('tests/%s' % self.name)

        logging.basicConfig(
                filename='tests/%s/%s.log' % (self.name, self.name),
                level=logging.DEBUG)

        self.tokenizer = self.tokenizer()

        logging.info("Building testing set")
        self.testing_documents = self.documents(self.testing_set())
        logging.info("Testing set builded")

        logging.info("Building training set")
        self.training_documents = self.documents(self.training_set())
        logging.info("Training set builded")

        logging.info("Initializing model")
        self.model = self.model(self.training_documents)

        logging.info("Generating table of test results")
        self.dump(*self.table_test(),
                output='tests/%s/test_result.log' % self.name)

        logging.info("Generating table of model stats")
        self.dump(*self.table_model_stats(),
                output='tests/%s/model_stats.log' % self.name)

        logging.info("Generating table of features stats in training set")
        self.dump(*self.table_features_stats(self.training_documents),
                output='tests/%s/training_features_stats.log' % self.name,
                order_by=0, reverse=True)

        logging.info("Generating table of features stats in testing set")
        self.dump(*self.table_features_stats(self.testing_documents),
                output='tests/%s/testing_features_stats.log' % self.name,
                order_by=0, reverse=True)

        logging.info("Generating table of documents")
        self.dump(*self.table_documents(),
                output='tests/%s/documents.log' % self.name,
                order_by=0, reverse=True, uniq=0)

        logging.info("Generating table of terms")
        self.dump(*self.table_terms(),
                output='tests/%s/terms.log' % self.name,
                order_by=5, reverse=True)

    def documents(self, data_set):
        documents = [Document(
            self.tokenizer.tokens(row[0]),
            label=row[1],
            uniqid=row[2]) for row in data_set]
        return documents

    def table_test(self):
        rows = []
        header = ('LABEL', 'CORRECTS', 'INCORRECTS', 'PERCENTAGE')

        total = defaultdict(int)
        corrects = defaultdict(int)

        mod = len(self.testing_documents) / 100 * 10
        for i, document in enumerate(self.testing_documents):
            if document.label == self.model.classify(document):
                corrects[document.label] += 1
            total[document.label] += 1
            if not i % mod:
                percentage = i / len(self.testing_documents) * 100
                logging.info("%d%% label(%s) documents tested" % (percentage, document.label))

        for label in total.keys():
            incorrects = total[label] - corrects[label]
            percentage = corrects[label] / total[label] * 100
            rows.append((label, corrects[label], incorrects, percentage))

        logging.info("%.2f%% correct" % (sum(corrects.values()) / sum(total.values()) * 100))
        return (rows, header)

    def table_features_stats(self, documents):
        rows = []
        header = ('FEATURES', 'DOCUMENTS')
        counter = defaultdict(int)
        for document in documents:
            size = len(self.model._features.intersection(document))
            counter[size] += 1
        for size, count in counter.items():
            rows.append((size, count))
        return (rows, header)

    def table_terms(self):
        header = ['TERM']
        for label in self.model._scoring.labels():
            header.append('DC(%s)' % label)
            header.append('DF(%s)' % label)
        header.append('WEIGHT')
        rows = []
        for i, term in enumerate(self.model._scoring.terms()):
            row = []
            row.append(term)
            for label in self.model._scoring.labels():
                row.append(self.model._scoring.dc(term, label))
                row.append(self.model._scoring.df(term, label))
            row.append(self.model._d(term))
            rows.append(row)
        return (rows, header)

    def table_documents(self):
        rows = []
        header = ['#FEATURES']
        for label in self.model._scoring.labels():
            header.append('P(%s)' % label)
        header.append('LABEL')
        header.append('PREDICTION')
        header.append('ID')
        header.append('FEATURES')
        for document in self.testing_documents:
            row = []
            features = self.model._features.intersection(document.terms)
            row.append(len(features))
            for label in self.model._scoring.labels():
                row.append(self.model._p(document, label))
            row.append(document.label)
            row.append(self.model.classify(document))
            row.append(document.uniqid)
            row.append(features)
            rows.append(row)
        return (rows, header)

    def table_model_stats(self):
        header = ['TERMS', 'FEATURES']
        row = []
        for label in self.model._scoring.labels():
            header.append('DOCUMENTS(%s)' % label)
            row.append(self.model._scoring.dc(label=label))
        row.append(len(self.model._scoring.terms()))
        row.append(len(self.model._features))
        return ([row], header)

    def dump(self, rows, header=None, output=sys.stdout,
            order_by=0,
            truncate=None,
            reverse=True,
            uniq=0,
            formatter=None):
        duplicates = 0
        prev_row = None
        rows = sorted(rows, key=lambda row: row[order_by], reverse=reverse)
        total = len(rows)
        rows = rows[:truncate]
        output = open(output, 'w')
        if not formatter:
            formatter = lambda field: '%010s' % (
                    round(field, 2) if type(field) is float else field)
        if header:
            rows = [header] + rows
        for i, row in enumerate(rows):
            if row[uniq:] == prev_row and i < len(rows) - 1:
                duplicates += 1
                continue
            if duplicates:
                output.write(" (...) %d times" % duplicates)
                duplicates = 0
            i and output.write('\n')
            output.write('\t'.join([formatter(field) for field in row]))
            prev_row = row[uniq:]
        output.write("\nTotal = %d rows" % total)
        output.close()


if __name__ == "__main__":
    repository = Repository()

    test_battery = []

    test = Test('test1')
    test.tokenizer = lambda: Tokenizer()
    test.model = lambda training_documents: Model(training_documents, 500)
    test.training_set = lambda: repository.get(
            selector=TRAINING,
            source=LIVRA,
            limit=3000,
            settings=['c.creation_date > now() - interval \'3 year\''])
    test.testing_set = lambda: repository.get(
            selector=TESTING,
            source=LIVRA,
            limit=100,
            settings=['c.creation_date < now() - interval \'1 year\''])
    test_battery.append(test)

    test = Test('test2')
    test.tokenizer = lambda: Tokenizer()
    test.model = lambda training_documents: Model(training_documents, 500)
    test.training_set = lambda: repository.get(selector=TRAINING, source=ENRON, limit=1500)
    test.testing_set = lambda: repository.get(selector=TESTING, source=ENRON, limit=1000)
    test_battery.append(test)

    test = Test('test3')
    test.tokenizer = lambda: Tokenizer()
    test.model = lambda training_documents: Model(training_documents, 500)
    test.training_set = lambda: repository.get(selector=TRAINING, source=LIVRA, limit=3000)
    test.testing_set = lambda: repository.get(selector=TESTING, source=LIVRA, limit=100)
    test_battery.append(test)

    for test in test_battery:
        test.run()
