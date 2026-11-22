"""The packaged word-frequency file is found and read (thorwhalen/idiom#8)."""

from idiom import most_frequent_words, word_frequency_posixpath


def test_packaged_frequency_file_exists():
    assert word_frequency_posixpath.is_file()


def test_most_frequent_words_returns_counts_most_frequent_first():
    words = most_frequent_words(max_n_words=10)
    assert len(words) == 10
    counts = list(words.values())
    assert counts == sorted(counts, reverse=True)
