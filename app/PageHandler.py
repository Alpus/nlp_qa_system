from . import HTMLParser, HTMLPageParser
import random

class PageHandler:
    def __init__(self, url, min_paragraph_length=100):
        self.min_paragraph_length = min_paragraph_length

        self.parser = HTMLPageParser(url)
        self.parser.parse()
        self.pages = []
        for filename, text in self.parser.iterate_over_texts():
            self.pages.append(text.lower())

    def get_paragraphs(self, text):
        paragraphs = list(filter(lambda x: len(x) >= self.min_paragraph_length,
                            map(lambda x: x.strip(), text.split('\n'))))
        return paragraphs

    def __getitem__(self, item):
        return self.get_paragraphs(self.pages[item])

    def __len__(self):
        return len(self.pages)