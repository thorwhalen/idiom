# idiom

Access to wordvec data and useful functions that use it.

### Module Attributes

| [`idf`](#idiom.idf)   | factor[word] = log(N / (1 + count[word])), where N is the total number of documents in the corpus.   |
|--------------------------------------------------------|------------------------------------------------------------------------------------------------------|

### Functions

| [`closest_words`](#idiom.closest_words)(word[, k, search_words, ...])   | Search words related to a give word.                            |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `cosine`(x, y)                                                                                 |                                                                 |
| `english_word2vec_stream`([word_filt, ...])                                                    |                                                                 |
| `get_english_word2vec_zip_filepath`()                                                          |                                                                 |
| `get_singular`(word)                                                                           |                                                                 |
| [`get_singular_w_inflect`](#idiom.get_singular_w_inflect)(word)                  | Get the singular form of a word using inflect                   |
| [`get_singular_w_nltk`](#idiom.get_singular_w_nltk)(word)                     | Get the singular form of a word using nltk's wordnet            |
| [`get_vec_of_word`](#idiom.get_vec_of_word)([corpus])                     | Get a word_2_vec dict given an alias name                       |
| `line_to_raw_word_vec`(line[, encoding, errors])                                               |                                                                 |
| `mk_tokenizer`(tokenizer)                                                                      |                                                                 |
| [`most_frequent_words`](#idiom.most_frequent_words)([max_n_words])            | The list of most frequent words ordered by most frequent first. |
| [`most_frequent_words_set`](#idiom.most_frequent_words_set)([max_n_words])        | The set of most frequent words.                                 |
| `vec_of_word_default_factory`()                                                                |                                                                 |
| `word_and_vecs`(fp)                                                                            |                                                                 |
| `word_count_df`()                                                                              |                                                                 |
| [`word_frequencies`](#idiom.word_frequencies)()                            | The set of most frequent words.                                 |
| `word_frequency_streams_of_zip`()                                                              |                                                                 |

### Classes

| [`SearchOld`](#idiom.SearchOld)([wordvec_zip_filepath, ...])      |                       |
|----------------------------------------------------------------------------------------------|-----------------------|
| [`StreamsOfZip`](#idiom.StreamsOfZip)(zip_file[, prefix, open_kws])  |                       |
| [`WordRawVecCreek`](#idiom.WordRawVecCreek)(stream)                     |                       |
| [`WordVec`](#idiom.WordVec)([vec_of_word])                      | Terms:                |
| [`WordVecCreek`](#idiom.WordVecCreek)(stream[, word_filt])           |                       |
| [`WordVecSearch`](#idiom.WordVecSearch)([word_vec, n_neighbors])      | Make a search engine. |
| [`WordVecsOfZip`](#idiom.WordVecsOfZip)(zip_file[, prefix, open_kws]) |                       |

### *class* idiom.SearchOld(wordvec_zip_filepath=None, search_words=None, wordvec_name_in_zip='wiki-news-300d-1M-subword.vec', n_neighbors=37, verbose=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

### Example

```text
zip_filepath = '/D/Dropbox/_odata/misc/wiki-news-300d-1M-subword.vec.zip'

import pandas as pd
df = pd.read_excel('/Users/twhalen/Downloads/pypi package names.xlsx')
target_words = set(df.word)

from grub.examples.pypi import Search

s = Search(zip_filepath, search_words=target_words)
s.search('search for the right name')
```

#### tokenizer(string, pos=0, endpos=9223372036854775807)

Return a list of all non-overlapping matches of pattern in string.

### *class* idiom.StreamsOfZip(zip_file, prefix='', open_kws=None)

Bases: `FileStreamsOfZip`

### *class* idiom.WordRawVecCreek(stream)

Bases: `Creek`

#### *static* data_to_obj(line, encoding='utf-8', errors='strict')

Return `x` unchanged.

#### *static* pre_iter(instance, , n=1)

Skip the first `n` items of `instance`.

### *class* idiom.WordVec(vec_of_word=vec_of_word_default_factory())

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

Terms:

> - query is anything (usually a string) that can be fed to tokenizer to product a list of tokens
> - words are tokens that are contained in the vec_of_word Mapping (keys)
> - vec is a vector (from the same space as vec_of_word values): Used as a finger print of a query
>   : (through it’s tokenization and a formula on the corresponding word’s vectors)

### Notes

- If using a non-dict vec_of_word mapping, make sure the \_\_contains_\_ is efficient, or processing will be slow.

```text
w = WordVec(vec_of_word=wv)
assert w.dist('france capital', 'paris') < w.dist('france capital', 'rome')
assert w.dist('italy capital', 'rome') < w.dist('italy capital', 'paris')
```

#### dist(q1, q2)

Cosine distance between two queries (through their corresponding vectors)

#### query_to_vec(query)

The vector computed from a query (process is query->tokens->words->vecs->mean).
Note that a query that leads to an empty list of words will result in the mean of all vectors in vec_to_word

### *class* idiom.WordVecCreek(stream, word_filt=None)

Bases: `Creek`

#### *static* data_to_obj(wv)

Return `x` unchanged.

#### pre_iter(stream)

Return `x` unchanged.

### *class* idiom.WordVecSearch(word_vec=None, n_neighbors=10, \*\*knn_kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Make a search engine.
Trains on a corpus of vectors (or {word: vector,…} mapping

#### fit(corpus=None)

Fit on the given corpus

* **Parameters:**
  **corpus** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping), [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – 

  A Mapping or iterable whose values we will fit on
  - corpus values must be valid self.word_vec arguments (usually strings)
  - corpus keys (or indices, if corpus wasn’t a Mapping) will be used in results of search
  - if not specified, will default to word_vec keys

### *class* idiom.WordVecsOfZip(zip_file, prefix='', open_kws=None)

Bases: `Store`

### idiom.closest_words(word, k=10, search_words=None, vec_of_word=None)

Search words related to a give word.
Given a word, search for the `k` closest words to it from a search corpus
(which may just be the wordvec words filtered for specific patterns).

For example, find the closest 10 words to ‘mad’ that start with an L.

```pycon
>>> starts_with_L = lambda x: x.startswith('l')
>>> closest_words('mad', k=10, search_words=starts_with_L)
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
```

Recipe: To avoid errors when giving an explicit list, you may want to filter out
those words that wordvecs doesn’t have:

```text
search_words = filter(lambda w: w in wordvecs, search_words)
```

### idiom.get_singular_w_inflect(word)

Get the singular form of a word using inflect

#### SEE ALSO
[`get_singular_w_nltk()`](#idiom.get_singular_w_nltk), which creates plurals through a different method,
(a semantic approach). Once the data is loaded, nltk is faster than the inflect,
version, but gives less accurate results, since it’s idea of singular is “the
base concept”.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### idiom.get_singular_w_nltk(word)

Get the singular form of a word using nltk’s wordnet

#### SEE ALSO
[`get_singular_w_inflect()`](#idiom.get_singular_w_inflect), which creates plurals through a different method,
(a rule-based approach). Once the data is loaded, nltk is faster than inflect,
but gives less accurate results, since it’s idea of singular is “the base concept”.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### idiom.get_vec_of_word(corpus='most_frequent_english')

Get a word_2_vec dict given an alias name

### idiom.idf *= <idiom._IDF object>*

factor[word] = log(N / (1 + count[word])), where N is the total number of documents in the corpus. This formula is a commonly used variant of IDF in the TF-IDF approach. The logarithmic term ensures that words with very high counts are not penalized too heavily, while still giving a significant weight to words that occur in a small subset of documents.
Smoothed IDF: factor[word] = log((N + 1) / (1 + count[word])) + 1, where N is the total number of documents in the corpus. This formula is similar to the logarithmic IDF, but with a smoothing term of 1 added to the numerator and denominator. This helps to avoid division by zero when a word appears in all documents, and gives a non-zero weight to all words.
Probabilistic IDF: factor[word] = log((N - count[word] + 0.5) / (count[word] + 0.5)), where N is the total number of documents in the corpus. This formula is a variant of IDF that takes into account the document frequency of a word, as well as the collection frequency. The term 0.5 is added to the numerator and denominator to avoid division by zero.
Max IDF: factor[word] = log(N / max(count.values())) - log(1 + count[word]), where N is the total number of documents in the corpus. This formula gives a weight to words based on their inverse document frequency, with a term subtracted to penalize words that occur frequently in a single document. The denominator of the second logarithm ensures that words with zero counts are not penalized too heavily.

* **Type:**
  Logarithmic IDF

### idiom.most_frequent_words(max_n_words=100000)

The list of most frequent words ordered by most frequent first.

#### NOTE
Twice faster than using FilesOfZip and pandas.read_csv

### idiom.most_frequent_words_set(max_n_words=100000)

The set of most frequent words.

#### NOTE
Twice faster than using FilesOfZip and pandas.read_csv

### idiom.word_frequencies()

The set of most frequent words.

#### NOTE
Twice faster than using FilesOfZip and pandas.read_csv

### Modules

| [`high_dims`](idiom.high_dims.html.md#module-idiom.high_dims)   | Tools to study high-dimensional statistics.   |
|-------------------------------------------------------------------------------------|-----------------------------------------------|
