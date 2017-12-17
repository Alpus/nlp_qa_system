import os
import pickle
import re

from bs4 import BeautifulSoup
from urllib import request
from urllib.parse import urljoin, urlparse

from .settings.paths import TEXTS_DIR
from .settings import logger


class HTMLParser:
    def __init__(self, main_url):
        self._urls = []
        self._texts = []
        self._main_url = main_url

        if urlparse(main_url).scheme == '':
            self._main_url = 'https://' + self._main_url

        logger.info('Initialize HTMLParser with "{}"'.format(self._main_url))

    def _traverse(self, from_url=None):
        if from_url is None:
            from_url = self._main_url

        links = BeautifulSoup(request.urlopen(from_url)).find_all('a')
        for link in links:
            href = link.get('href')
            if href.startswith('#'):
                href = '/' + href
            url = urljoin(from_url, href)
            url = urlparse(url).geturl()
            is_external = url.startswith('http') and not url.startswith(self._main_url)
            if not is_external and not url in self._urls:
                yield url
                self._traverse(url)

    def _get_text_filename(self, url, i):
        url_str = url.replace('https://', '').replace('#', '').replace('.', '_').replace('/', '_').replace('-', '_')
        if i > 9999:
            logger.warn('paragraph list is too long')
        return '{}_{:05d}'.format(url_str, i)

    def _save_text(self, text, path_to_save):
        with open(path_to_save, 'wb') as file:
            pickle.dump(text, file)
            logger.info('Save texts to \"{}\"'.format(path_to_save))

    def parse(self, path_to_save=TEXTS_DIR):
        try:
            os.mkdir(path_to_save)
        except:
            logger.info('Folder "{}" already exists'.format(path_to_save))
            self.num_texts = len(os.listdir(path_to_save))
            return

        self.num_texts = 0
        for i, url in enumerate(self._traverse()):
            logger.info('({}) {}'.format(i, url))
            try:
                soup = BeautifulSoup(request.urlopen(url))

                # kill all script and style elements
                for script in soup(["script", "style"]):
                    script.extract()  # rip it out

                # get text
                text = soup.get_text()

                # replace eol, tabs, multiple whitespaces)
                text = re.sub('\\t', ' ', text)
                text = re.sub(' +', ' ', text)
                text = re.sub(' \n', '\n', text)

                # remove text in parentheses
                text = re.sub(r'\{[^}]*\}', '', text)

                # remove html comments
                # text = re.sub(re.compile("/\*.*?\*/",re.DOTALL ), "", text)
                # text = re.sub(re.compile("//.*?\n" ), "", text)

                # split to paragraphs of length >= 100
                paragraphs = filter(lambda x: len(x) >= 100,
                                    map(lambda x: x.strip(), text.split('\n\n')))
                for i, p in enumerate(paragraphs):
                    p = p.strip()
                    # print('PARAGRAPH {}\n[ {} ]\n\n'.format(i + 1, p))
                    self._save_text(p, path_to_save=os.path.join(path_to_save, self._get_text_filename(url, i)))
                    self.num_texts += 1
            except:
                logger.warn('Invalid url: {}'.format(url))

    def __iter__(self):
        return self.iterate_over_texts()

    def iterate_over_texts(self, texts_dir=TEXTS_DIR):
        logger.info('Iterate over {} saved texts'.format(self.num_texts))

        for filename in sorted(filter(lambda x: not x.startswith('.'), os.listdir(texts_dir))):
            with open(os.path.join(texts_dir, filename), 'rb') as file:
                yield (filename, pickle.load(file))
