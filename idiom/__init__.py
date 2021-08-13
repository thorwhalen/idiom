"""
Access to wordvec data and useful functions that use it.
"""
from functools import cached_property, partial, lru_cache
import re
from typing import Mapping, Any, Callable, Union, Iterable, Optional
from importlib_resources import files as package_files
from dataclasses import dataclass, field
from itertools import islice
from heapq import nlargest
from typing import Iterable, Callable, Union, Optional


import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError

from py2store import Store
from py2store.slib.s_zipfile import FileStreamsOfZip, FilesOfZip

# from py2store.base import Stream

from creek import Creek
from creek.util import PreIter

data_files = package_files("idiom.data")

english_word2vec_url = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-english/"
    "wiki-news-300d-1M-subword.vec.zip"
)
word_frequency_posixpath = data_files.joinpath("english-word-frequency.zip")


def closest_words(
    word, k=10, search_words: Optional[Union[Callable, Iterable]] = None, vec_of_word=None
):
    """Search words related to a give word.
    Given a word, search for the `k` closest words to it from a search corpus
    (which may just be the wordvec words filtered for specific patterns).

    For example, find the closest 10 words to 'mad' that start with an L.

    >>> starts_with_L = lambda x: x.startswith('l')
    >>> closest_words('mad', k=10, search_words=starts_with_L)  # doctest: +SKIP
    ['lunatic',
     'loony',
     'loather',
     'loathesome',
     'love-sick',
     'loooove',
     'lovestruck',
     'lovesick',
     'luddite',
     'lazy-minded']

    Recipe: To avoid errors when giving an explicit list, you may want to filter out
    those words that wordvecs doesn't have:

    ```
    search_words = filter(lambda w: w in wordvecs, search_words)
    ```

    """
    vec_of_word = get_vec_of_word(vec_of_word)
    target_word_vector = vec_of_word[word]
    if search_words is None:
        search_words = vec_of_word  # everything we have in vec_of_word
    elif isinstance(search_words, Callable):
        words_filter_func = search_words
        search_words = filter(words_filter_func, vec_of_word)
    assert isinstance(search_words, Iterable), (
        "search_words should None, an iterable or a filter " "function"
    )

    search_word_vectors = map(lambda k: (k, vec_of_word[k]), search_words)
    return [
        y[0]
        for y in nlargest(
            k, search_word_vectors, key=lambda x: -cosine(target_word_vector, x[1])
        )
    ]


@lru_cache(maxsize=1)
def most_frequent_words(max_n_words=100_000):
    """The set of most frequent words.
    Note: Twice faster than using FilesOfZip and pandas.read_csv
    """
    z = FileStreamsOfZip(str(word_frequency_posixpath))
    with z["unigram_freq.csv"] as zz:
        return set([x.decode().split(",")[0] for x in islice(zz, 0, max_n_words)])


def get_english_word2vec_zip_filepath():
    from graze import Graze

    g = Graze()
    if english_word2vec_url not in g:
        print(
            f"Downloading {english_word2vec_url} and storing it locally "
            f"(in {g.filepath_of(english_word2vec_url)})"
        )

    zip_filepath = g.filepath_of(english_word2vec_url)

    return zip_filepath


def line_to_raw_word_vec(line, encoding="utf-8", errors="strict"):
    word, vec = line.split(maxsplit=1)
    return word.decode(encoding, errors), vec


skip_one_item = partial(PreIter().skip_items, n=1)


class WordRawVecCreek(Creek):
    pre_iter = staticmethod(skip_one_item)
    data_to_obj = staticmethod(line_to_raw_word_vec)


class WordVecCreek(Creek):
    def __init__(self, stream, word_filt=None):
        super().__init__(stream)
        if not callable(word_filt):
            word_filt = partial(filter, word_filt)
        self.word_filt = word_filt

    def pre_iter(self, stream):
        next(stream)  # consume the first line (it's a header)
        return filter(
            lambda wv: self.word_filt(wv[0]), map(line_to_raw_word_vec, stream)
        )  # split word and vec

    data_to_obj = staticmethod(lambda wv: (wv[0], np.fromstring(wv[1], sep=" ")))


class WordVecsOfZip(Store.wrap(FileStreamsOfZip)):
    _obj_of_data = staticmethod(WordVecCreek)


def english_word2vec_stream(
    word_filt=None, zip_filepath=None, key="wiki-news-300d-1M-subword.vec"
):
    zip_filepath = zip_filepath or get_english_word2vec_zip_filepath()
    lines_of_zip = FileStreamsOfZip(zip_filepath)[key]
    return WordVecCreek(lines_of_zip, word_filt)


def word_and_vecs(fp):
    #     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

    # consume the first line (n_lines, n_dims) not yielded
    # n_lines, n_dims = map(int, fp.readline().decode().split())
    for line in fp:
        tok, *vec = line.decode().rstrip().split(" ")
        yield tok, tuple(map(float, vec))


Vec = np.ndarray
VecStore = Mapping[Any, Vec]
WordVecStore = Mapping[str, Vec]


def mk_tokenizer(tokenizer):
    tokenizer_spec = tokenizer
    if isinstance(tokenizer_spec, (str, re.Pattern)):
        pattern = re.compile(tokenizer_spec)

        def tokenizer(string: str):
            return pattern.findall(string)

    return tokenizer


alpha_num_p = re.compile(r"[\w-]+")
letters_p = re.compile(r"[a-z]+")


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


@lru_cache(maxsize=1)
def vec_of_word_default_factory():
    words = most_frequent_words()
    return dict(english_word2vec_stream(word_filt=words.__contains__))


@lru_cache(maxsize=1)
def _get_vec_of_word(corpus):
    if isinstance(corpus, str):
        if corpus == "most_frequent_english":
            words = most_frequent_words()
            return dict(english_word2vec_stream(word_filt=words.__contains__))
        elif corpus == "english_all":
            dict(english_word2vec_stream())
    raise ValueError(f"Unrecognized corpus value: {corpus}")


DFLT_WORDVEC_CORPUS_NAME = "most_frequent_english"

word_to_vec_corpus_aliases = {
    "most_frequent_english": "most_frequent_english",
    "most_frequent": "most_frequent_english",
    "english_70982": "most_frequent_english",
    "english_all": "english_all",
    "english_999994": "english_all",
}


def get_vec_of_word(corpus=DFLT_WORDVEC_CORPUS_NAME):
    """Get a word_2_vec dict given an alias name"""
    if corpus is None:
        corpus = DFLT_WORDVEC_CORPUS_NAME
    if isinstance(corpus, str):
        if corpus in {"most_frequent_english", "most_frequent", "english_70982"}:
            _get_vec_of_word("most_frequent_english")
        elif corpus in {"english_all", "english_999994"}:
            _get_vec_of_word("most_frequent_english")
    raise ValueError(f"Unrecognized corpus value: {corpus}")


@dataclass
class WordVec(Mapping):
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

    vec_of_word: WordVecStore = field(
        default_factory=vec_of_word_default_factory, repr=False
    )
    tokenizer = r"[\w-]+"

    def __post_init__(self):
        self.tokenizer = mk_tokenizer(self.tokenizer)

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
            vectors = list(filter(lambda v: v is not None, self.vec_matrix(words)))
            if len(vectors) > 0:
                return np.mean(vectors, axis=0)
        return self.global_mean  # if all else fails

    def query_2_words(self, query):
        return [tok for tok in self.tokenizer(query) if tok in self.vec_of_word]

    @cached_property
    def global_mean(self):
        return np.sum(list(self.vec_of_word.values()), axis=0)
        # return np.mean(list(self.vec_of_word.values()), axis=0)

    def vec_matrix(self, words):
        return np.array([self.vec_of_word.get(w, None) for w in words])

    def __repr__(self):
        tokenizer_name = getattr(self.tokenizer, "__name__", "unnamed_tokenizer")
        return (
            f"{self.__class__.__name__}("
            f"vec_of_word={type(self.vec_of_word).__name__} with {len(self.vec_of_word)} words, "
            f"tokenizer={tokenizer_name})"
        )

    __call__ = query_to_vec

    # TODO: Replace with "explicit" decorator
    def __getitem__(self, k):
        return self.vec_of_word[k]

    def __len__(self):
        return len(self.vec_of_word)

    def __contains__(self, k):
        return k in self.vec_of_word

    def __iter__(self):
        return iter(self.vec_of_word)


Corpus = Optional[Union[Mapping, Iterable]]


class WordVecSearch:
    """Make a search engine.
    Trains on a corpus of vectors (or {word: vector,...} mapping
    """

    corpus_ = None
    corpus_keys_array_ = None

    def __init__(self, word_vec: WordVec = None, n_neighbors=10, **knn_kwargs):
        """

        :param word_vec: A WordVec object that will
        :param n_neighbors:
        :param knn_kwargs:
        """
        word_vec = word_vec or WordVec()
        self.word_vec = word_vec
        knn_kwargs = dict(n_neighbors=n_neighbors, metric="cosine", **knn_kwargs)
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
        except NotFittedError:
            self._when_searched_on_unfit_instance()
            self.search(query, include_dist)

    def _when_searched_on_unfit_instance(self):
        from warnings import warn

        warn(
            "The search object wasn't fitted yet, so I'm fitting it on the "
            "wordvec data itself. "
            "To avoid this message, do a .fit() before using the search "
            "functionality."
        )
        return self.fit()

    __call__ = search


class StreamsOfZip(FileStreamsOfZip):
    def _obj_of_data(self, data):
        return line_to_raw_word_vec(data)


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

    tokenizer = re.compile(r"\w+").findall

    def __init__(
        self,
        wordvec_zip_filepath=None,
        search_words=None,
        wordvec_name_in_zip="wiki-news-300d-1M-subword.vec",
        n_neighbors=37,
        verbose=False,
    ):
        self.wordvec_zip_filepath = (
            wordvec_zip_filepath or get_english_word2vec_zip_filepath()
        )
        self.wordvec_name_in_zip = wordvec_name_in_zip
        if search_words:
            search_words = set(search_words)
        self.search_words = search_words
        self.n_neighbors = n_neighbors
        self.verbose = verbose

    @cached_property
    def stream(self):
        return StreamsOfZip(self.wordvec_zip_filepath)

    @cached_property
    def wordvecs(self):
        if self.verbose:
            print("Gathering all the word vecs. This could take a few minutes...")
        with self.stream[self.wordvec_name_in_zip] as fp:
            all_wordvecs = dict(word_and_vecs(fp))
        return all_wordvecs

    def filtered_wordvecs(self, tok_filt):
        with self.stream[self.wordvec_name_in_zip] as fp:
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
        target_wv = dict(self.filtered_wordvecs(lambda x: x in self.search_words))
        X = np.array(list(target_wv.values()))

        knn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="cosine").fit(X)
        knn.words = np.array(list(target_wv.keys()))
        return knn

    def search(self, query):
        query_vec = self.query_to_vec(query)
        r_dist, r_idx = self.knn.kneighbors(query_vec.reshape(1, -1))
        return self.knn.words[r_idx]
