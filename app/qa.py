from abc import ABCMeta, abstractmethod

import numpy as np
from gensim.models import KeyedVectors
from scipy.spatial import KDTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sortedcontainers import SortedList
from tqdm import tqdm

from . import HTMLParser, RuTokenizer, MystemTokenizer
from .settings.paths import WORD2VEC_WEIGHTS_FILE


class QA(metaclass=ABCMeta):
    @abstractmethod
    def n_closest_paragraphs(self, question, n=1):
        """
        Args:
            question:
            n:

        Returns:
            list: n best texts
        """

        pass


class RUQA(QA):
    def __init__(self, url):
        self.parser = HTMLParser(url)
        self.parser.parse()
        self.tokenizer = RuTokenizer(annotators={'lemma', 'pos', 'ner'})


class FrequencyQA(RUQA):
    def _calculate_overlap(self, paragraph_tokens, question_tokens):
        paragraph_set = set(paragraph_tokens.words())
        question = set(question_tokens.words())
        return len(paragraph_set & question)

    def n_closest_paragraphs(self, question, n=1):
        question_tokens = self.tokenizer(question)

        best_n_weight_paragraph = SortedList()

        for filename, text in self.parser:
            paragraph_tokens = self.tokenizer(text)
            overlap = self._calculate_overlap(paragraph_tokens,
                                              question_tokens)

            best_n_weight_paragraph.add((-overlap, text))

            if len(best_n_weight_paragraph) > n:
                best_n_weight_paragraph.pop()

        return [text for weight, text in best_n_weight_paragraph]


class UrlTfIdfQA(RUQA):
    def __init__(self, url):
        super().__init__(url)
        self._texts = []
        for filename, text in self.parser.iterate_over_texts():
            self._texts.append(text.lower())

        self._tfIdf = TfidfVectorizer()
        self._tokens = self._tfIdf.fit_transform(self._texts)
        self._word_dict = self._tfIdf.vocabulary_

    def _build_one_hot_vector_by_string_(self, string):
        description = np.zeros(len(self._word_dict))
        for word in string.split():
            if word in self._word_dict:
                description[self._word_dict[word]] = 1
        return description

    def _get_best_indexes(self, question, n):
        oh = self._build_one_hot_vector_by_string_(question)
        scores = (self._tokens * oh)

        return scores.argsort()[-n:][::-1]

    def n_closest_paragraphs(self, question, n=1):
        return np.array(self._texts)[self._get_best_indexes(question, n)]


class W2VQA(RUQA):
    def __init__(self, url):
        super().__init__(url)
        self._w2v = KeyedVectors.load_word2vec_format(
            WORD2VEC_WEIGHTS_FILE, binary=True)
        self.tokenizer = MystemTokenizer()

        self._texts = []
        self._centroids = []
        for filename, text in tqdm(self.parser.iterate_over_texts()):
            centroid = self.calculate_centroid(text.lower())
            if centroid is None:
                continue
            self._texts.append(text.lower())
            self._centroids.append(centroid)
        self._centroids = KDTree(self._centroids)

    def calculate_centroid(self, text):
        center = 0
        num = 0
        words = self.tokenizer(text).words()
        for word in words:
            if word in self._w2v:
                center += self._w2v[word]
                num += 1
        if num == 0:
            return None
        return center / num

    def calculate_wmd(self, question, text):
        q_words = self.tokenizer(question).words()
        t_words = self.tokenizer(text).words()

        def filter_fn(word):
            return word in self._w2v

        q_words = list(filter(filter_fn, q_words))
        t_words = list(filter(filter_fn, t_words))
        return self._w2v.wmdistance(q_words, t_words)

    def n_closest_paragraphs(self, question, n=1, mode='wmd', n_filter=20):
        if mode == 'centroids':
            q_centroid = self.calculate_centroid(question)
            _, inds = self._centroids.query(q_centroid, k=n)
            res = []
            for ind in inds:
                res.append(self._texts[ind])
        elif mode == 'wmd':
            q_centroid = self.calculate_centroid(question)
            _, inds = self._centroids.query(q_centroid, k=n_filter)
            texts_dist = []
            for ind in inds:
                texts_dist.append((self._texts[ind],
                                   self.calculate_wmd(question,
                                                      self._texts[ind])))
            texts_dist.sort(key=lambda text_dist: text_dist[1])
            return [text for text, dist in texts_dist[:n]]
        else:
            raise NotImplementedError("mode {} is not supported".format(mode))

        return res


class TfIdfQA:
    def __init__(self, texts):
        self.texts = texts
        self._tfIdf = TfidfVectorizer()
        self._tokens = self._tfIdf.fit_transform(self.texts)
        self._word_dict = self._tfIdf.vocabulary_

    def _build_one_hot_vector_by_string_(self, string):
        description = np.zeros(len(self._word_dict))
        for word in string.split():
            if word in self._word_dict:
                description[self._word_dict[word]] = 1
        return description

    def get_best_indexes(self, question, n):
        oh = self._build_one_hot_vector_by_string_(question)
        scores = (self._tokens * oh)
        scores[scores == 0] = -1e9
        return scores.argsort()[-n:][::-1]

    def n_closest_paragraphs(self, question, n=1):
        return np.array(self.texts)[self.get_best_indexes(question, n)]


class TfIdf2stepQA:
    def __init__(self, pageHandler, n_best_pages=1):
        self.pageHandler = pageHandler
        self.n_best_pages = n_best_pages
        self.tokenizer = RuTokenizer(annotators={'lemma', 'pos', 'ner'})
        self.pageQA = TfIdfQA(self.get_tokenized_text(pageHandler.pages))


    def get_tokenized_text(self, texts):
        tokenized_texts = []
        for text in texts:
            tokenized_texts.append(' '.join(self.tokenizer(text).words(True)))
        return tokenized_texts

    def n_closest_paragraphs(self, question, n=1):
        question = ' '.join(self.tokenizer(question).words(True))
        best_indx = self.pageQA.get_best_indexes(question, self.n_best_pages)
        best_pages = []
        for i in best_indx:
            best_pages.extend(self.pageHandler[i])
        paragQA = TfIdfQA(self.get_tokenized_text(best_pages))
        indxes = paragQA.get_best_indexes(question, n)
        return np.array(best_pages)[indxes]
