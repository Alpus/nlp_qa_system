from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sortedcontainers import SortedList

from . import HTMLParser, RuTokenizer


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
            overlap = self._calculate_overlap(paragraph_tokens, question_tokens)

            best_n_weight_paragraph.add((-overlap, text))

            if len(best_n_weight_paragraph) > n:
                best_n_weight_paragraph.pop()

        return [text for weight, text in best_n_weight_paragraph]


class TfIdfQA(RUQA):
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
