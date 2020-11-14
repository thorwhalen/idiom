from functools import cached_property
import re
from typing import Mapping, Any, Callable, Union, Iterable, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError

from py2store.slib.s_zipfile import FileStreamsOfZip
from py2store.base import Stream


def line_to_raw_word_vec(line):
    word, vec = line.split(maxsplit=1)
    return word.decode(), vec


class WordVecStream(Stream):
    _obj_of_data = line_to_raw_word_vec


class StreamsOfZip(FileStreamsOfZip):
    def _obj_of_data(self, data):
        return line_to_raw_word_vec(data)


def word_and_vecs(fp):
    #     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

    # consume the first line (n_lines, n_dims) not yielded
    n_lines, n_dims = map(int, fp.readline().decode().split())
    for line in fp:
        tok, *vec = line.decode().rstrip().split(' ')
        yield tok, tuple(map(float, vec))


Vec = np.ndarray
VecStore = Mapping[Any, Vec]
WordVecStore = Mapping[str, Vec]


def mk_tokenizer(tokenizer):
    tokenizer_spec = tokenizer
    if isinstance(tokenizer_spec, (str, re.Pattern)):
        pattern = re.compile(tokenizer_spec)

        def tokenizer(string: str): return pattern.findall(string)
    return tokenizer


alpha_num_p = re.compile(r'[\w-]+')
letters_p = re.compile(r'[a-z]+')


#
# def keys_and_vals_of_coll(coll):
#     if isinstance(coll, Mapping):
#         return zip(*coll.items())
#     else:
#         return zip(*enumerate(coll))

# class FakeMapping:
#     def __init__(self, coll):
#         self.coll = coll
#
#     def values(self):
#         if isinstance(coll, Mapping):
#             return coll.items()
#         else:
#             return enumerate(coll)
#
#     def keys(self):
#         if isinstance(coll, Mapping):
#             return coll.items()
#         else:
#             return enumerate(coll)


def cosine(x, y):
    return cosine_distances([x], [y])[0][0]


from py2store import Store


class WordVec:
    """

    Terms:
        - query is anything (usually a string) that can be fed to tokenizer to product a list of tokens
        - words are tokens that are contained in the vec_of_word Mapping (keys)
        - vec is a vector (from the same space as vec_of_word values): Used as a finger print of a query
            (through it's tokenization and a formula on the corresponding word's vectors)


    Notes:
        - If using a non-dict vec_of_word mapping, make sure the __contains__ is efficient, or processing will be slow.

    ```
    w = WordVec(vec_of_word=wv)
    assert w.dist('france capital', 'paris') < w.dist('france capital', 'rome')
    assert w.dist('italy capital', 'rome') < w.dist('italy capital', 'paris')
    ```

    """

    def __init__(self,
                 vec_of_word: WordVecStore,
                 tokenizer=r'[\w-]+'):
        """Provides  methods to computing a vectors from an arbitrary query, using tokenizer and vec_of_word mapping.

        :param vec_of_word:
        :param tokenizer:
        """
        self.vec_of_word = vec_of_word
        self.tokenizer = mk_tokenizer(tokenizer)

    def dist(self, q1, q2):
        """Cosine distance between two queries (through their corresponding vectors)"""
        return cosine(self.query_to_vec(q1), self.query_to_vec(q2))

    def query_to_vec(self, query):
        """The vector computed from a query (process is query->tokens->words->vecs->mean).
        Note that a query that leads to an empty list of words will result in the mean of all vectors in vec_to_word
        """
        words = self.query_2_words(query)
        return self.mean_vec(words)

    def query_to_vec_matrix(self, query):
        words = self.query_2_words(query)
        return self.vec_matrix(words)

    def mean_vec(self, words):
        if len(words) > 0:
            return np.mean(self.vec_matrix(words), axis=0)
        else:
            return self.global_mean

    def query_2_words(self, query):
        return [tok for tok in self.tokenizer(query) if tok in self.vec_of_word]

    @cached_property
    def global_mean(self):
        return np.sum(list(self.vec_of_word.values()), axis=0)
        # return np.mean(list(self.vec_of_word.values()), axis=0)

    def vec_matrix(self, words):
        return np.array([self.vec_of_word.get(w, None) for w in words])

    def __repr__(self):
        tokenizer_name = getattr(self.tokenizer, '__name__', 'unnamed_tokenizer')
        return f"{self.__class__.__name__}(" \
               f"vec_of_word={type(self.vec_of_word).__name__} with {len(self.vec_of_word)} words, " \
               f"tokenizer={tokenizer_name})"

    __call__ = query_to_vec


Corpus = Optional[Union[Mapping, Iterable]]


class WordVecSearch:
    corpus_ = None
    corpus_keys_array_ = None

    def __init__(self, word_vec: WordVec, n_neighbors=10, **knn_kwargs):
        self.word_vec = word_vec
        knn_kwargs = dict(n_neighbors=n_neighbors, metric='cosine', **knn_kwargs)
        self.knn = NearestNeighbors(**knn_kwargs)

    def fit(self, corpus: Corpus = None):
        """Fit on the given corpus

        :param corpus: A Mapping or iterable whose values we will fit on
            - corpus values must be valid self.word_vec arguments (usually strings)
            - corpus keys (or indices, if corpus wasn't a Mapping) will be used in results of search
            - if not specified, will default to word_vec keys

        """

        if corpus is None:  # if corpus is not given, use word_vec as the corpus
            words = self.word_vec.vec_of_word.keys()
            corpus = dict(zip(words, words))
        elif not isinstance(corpus, Mapping):
            corpus = dict(enumerate(corpus))

        self.corpus_ = corpus
        vecs = np.array(list(map(self.word_vec, self.corpus_)))
        self.knn.fit(vecs)
        self.corpus_keys_array_ = np.array(list(self.corpus_.keys()))
        return self

    def search(self, query, include_dist=False):
        try:
            query_vec = self.word_vec(query)
            r_dist, r_idx = self.knn.kneighbors(query_vec.reshape(1, -1))
            corpus_keys = self.corpus_keys_array_[r_idx]
            if not include_dist:
                return corpus_keys
            else:
                return corpus_keys, r_dist
        except NotFittedError:  # Note: Should we warn that we're defaulting?
            return self.fit().search(query, include_dist)

    __call__ = search


class SearchOld:
    """
    Example:

    ```
    zip_filepath = '/D/Dropbox/_odata/misc/wiki-news-300d-1M-subword.vec.zip'

    import pandas as pd
    df = pd.read_excel('/Users/twhalen/Downloads/pypi package names.xlsx')
    target_words = set(df.word)

    from grub.examples.pypi import Search

    s = Search(zip_filepath, search_words=target_words)
    s.search('search for the right name')
    ```
    """
    tokenizer = re.compile('\w+').findall

    def __init__(self,
                 wordvec_zip_filepath,
                 search_words,
                 wordvec_name_in_zip='wiki-news-300d-1M-subword.vec',
                 n_neighbors=37,
                 verbose=False
                 ):
        self.wordvec_zip_filepath = wordvec_zip_filepath
        self.wordvec_name_in_zip = wordvec_name_in_zip
        self.search_words = set(search_words)
        self.n_neighbors = n_neighbors
        self.verbose = verbose

    @cached_property
    def stream(self):
        return StreamsOfZip(self.wordvec_zip_filepath)

    @cached_property
    def wordvecs(self):
        if self.verbose:
            print('Gathering all the word vecs. This could take a few minutes...')
        with self.stream[self.wordvec_name_in_zip] as fp:
            all_wordvecs = dict(word_and_vecs(fp))
        return all_wordvecs

    def filtered_wordvecs(self, tok_filt):
        with self.stream['wiki-news-300d-1M-subword.vec'] as fp:
            yield from filter(lambda x: tok_filt(x[0]), word_and_vecs(fp))

    def vec_matrix(self, words):
        return np.array([self.wordvecs.get(w, None) for w in words])

    def mean_vec(self, words):
        return np.mean(self.vec_matrix(words), axis=0)

    def query_to_vec(self, query):
        tokens = self.tokenizer(query)
        return self.mean_vec(tokens)

    def query_to_vec_matrix(self, query):
        tokens = self.tokenizer(query)
        return self.vec_matrix(tokens)

    @cached_property
    def knn(self):
        taget_wv = dict(self.filtered_wordvecs(lambda x: x in self.search_words))
        X = np.array(list(taget_wv.values()))

        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='cosine').fit(X)
        knn.words = np.array(list(taget_wv.keys()))
        return knn

    def search(self, query):
        query_vec = self.query_to_vec(query)
        r_dist, r_idx = self.knn.kneighbors(query_vec.reshape(1, -1))
        return self.knn.words[r_idx]
