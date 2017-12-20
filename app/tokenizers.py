import copy
import re
from abc import ABCMeta, abstractmethod

import pymorphy2
from pymystem3 import Mystem
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from polyglot.text import Text


class Tokens:
    """A class to represent a list of tokenized text."""

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """
        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', '')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(metaclass=ABCMeta):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    @abstractmethod
    def __call__(self, text):
        raise NotImplementedError


class RuTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
        """

        self.annotators = copy.deepcopy(kwargs.get('annotators', set()))
        self.include_pos = {'pos'} & self.annotators
        self.include_lemma = {'lemma'} & self.annotators
        self.include_ner = {'ner'} & self.annotators
        self.morph = pymorphy2.MorphAnalyzer()
        self.wt = WhitespaceTokenizer()
        self.rt = RegexpTokenizer(r'\w+')

    def __call__(self, text):

        # We don't treat new lines as tokens.
        clean_text = text.replace('\n', ' ')

        # remove punctuation
        clean_text = ' '.join(self.rt.tokenize(clean_text))

        # split by whitespaces and get spans
        spans = list(self.wt.span_tokenize(clean_text))
        n = len(spans)

        data = []
        for i in range(n):
            start_idx, end_idx = spans[i]

            token = clean_text[start_idx:end_idx]

            start_ws = start_idx
            if i + 1 < n:
                end_ws = spans[i + 1][0]
            else:
                end_ws = start_idx + len(token)

            token_ws = clean_text[start_ws:end_ws]

            lemma, pos, ent_type = '', '', ''
            if self.include_pos or self.include_lemma:
                p = self.morph.parse(token)[0]
                if self.include_lemma:
                    lemma = p.normal_form
                if self.include_pos:
                    pos = p.tag.POS

            if self.include_ner:
                entities = Text(token, hint_language_code='ru').entities
                if len(entities):
                    ent_type = entities[0].tag

            data.append((token, token_ws, spans[i], pos, lemma, ent_type))

        return Tokens(data, self.annotators, opts={'non_ent': ''})


class MystemTokenizer:
    def __init__(self):
        self.mystem = Mystem()
        self.text = None

    def __call__(self, text):
        self.text = text
        return self

    def words(self, upos=False):
        words = []
        for token in self.mystem.analyze(self.text):
            if 'analysis' in token:
                if len(token['analysis']) != 0:
                    grapheme = re.match("^\w+",
                                        token['analysis'][0]['gr']).group(0)
                    lexeme = token['analysis'][0]['lex']
                else:
                    grapheme = 'S'
                    lexeme = token['text']
                if upos:
                    grapheme = self.convert_to_UPOS(grapheme)
                words.append('_'.join([lexeme, grapheme]))
        return words

    def convert_to_UPOS(self, grapheme):
        mapping = {
            'A': 'ADJ',
            'ADV': 'ADV',
            'ADVPRO': 'ADV',
            'ANUM': 'ADJ',
            'APRO': 'DET',
            'COM': 'ADJ',
            'CONJ': 'SCONJ',
            'INTJ': 'INTJ',
            'NONLEX': 'X',
            'NUM': 'NUM',
            'PART': 'PART',
            'PR': 'ADP',
            'S': 'NOUN',
            'SPRO': 'PRON',
            'UNKN': 'X',
            'V': 'VERB',
        }
        return mapping[grapheme]
