"""Offline tests: WordVec, WordVecSearch, closest_words and high_dims helpers on a toy vocabulary."""

import warnings

import numpy as np
import pytest

import idiom
from idiom import (
    WordVec,
    WordVecSearch,
    closest_words,
    cosine,
    line_to_raw_word_vec,
    mk_tokenizer,
    word_and_vecs,
)
from idiom import high_dims as hd

VECS = {
    "cat": np.array([1.0, 0.0, 0.0]),
    "dog": np.array([0.9, 0.1, 0.0]),
    "car": np.array([0.0, 1.0, 0.0]),
    "truck": np.array([0.1, 0.9, 0.0]),
}


def test_cosine_and_tokenizer():
    assert cosine([1, 0], [1, 0]) == pytest.approx(0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(1)
    tok = mk_tokenizer(r"[a-z]+")
    assert tok("ab-cd 12 ef") == ["ab", "cd", "ef"]
    f = lambda s: s.split()
    assert mk_tokenizer(f) is f


def test_line_and_stream_parsing():
    assert line_to_raw_word_vec(b"hello 1 2 3") == ("hello", b"1 2 3")
    assert list(word_and_vecs([b"a 1 2\n", b"b 3 4\n"])) == [
        ("a", (1.0, 2.0)),
        ("b", (3.0, 4.0)),
    ]


def test_closest_words(monkeypatch):
    monkeypatch.setattr(idiom, "get_vec_of_word", lambda v: v)
    assert closest_words("cat", 2, vec_of_word=VECS)[0] == "cat"
    assert closest_words("cat", 2, vec_of_word=VECS)[1] == "dog"
    got = closest_words("car", 2, search_words=["cat", "truck"], vec_of_word=VECS)
    assert got[0] == "truck"
    got = closest_words("car", 1, search_words=lambda w: w != "car", vec_of_word=VECS)
    assert got == ["truck"]


def test_wordvec():
    wv = WordVec(vec_of_word=VECS)
    assert len(wv) == 4 and "cat" in wv and sorted(wv) == sorted(VECS)
    assert wv["cat"] is VECS["cat"]
    assert wv.query_2_words("the cat and dog") == ["cat", "dog"]
    assert np.allclose(wv("cat dog"), [0.95, 0.05, 0])
    assert wv.query_to_vec_matrix("cat car").shape == (2, 3)
    assert wv.dist("cat", "dog") < wv.dist("cat", "car")
    # no known word: falls back to the global mean
    assert np.allclose(wv("zzz"), wv.global_mean)
    assert "4 words" in repr(wv)


def test_wordvec_search():
    s = WordVecSearch(WordVec(vec_of_word=VECS), n_neighbors=2).fit()
    keys = s("cat")
    assert keys[0][0] == "cat"
    keys, dist = s.search("car", include_dist=True)
    assert keys[0][0] == "car" and dist[0][0] == pytest.approx(0, abs=1e-9)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="WordVecSearch.fit maps word_vec over corpus keys, not values, contradicting its docstring",
)
def test_wordvec_search_fit_on_corpus_values():
    s = WordVecSearch(WordVec(vec_of_word=VECS), n_neighbors=1).fit(["cat", "car"])
    assert s("truck")[0][0] == 1  # corpus keys are indices


def test_search_unfit_warns():
    s = WordVecSearch(WordVec(vec_of_word=VECS), n_neighbors=2)
    with pytest.warns(UserWarning, match="fitted"):
        s.search("cat")


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="get_vec_of_word never returns: valid corpus aliases fall through to the ValueError",
)
def test_get_vec_of_word_alias(monkeypatch):
    monkeypatch.setattr(idiom, "_get_vec_of_word", lambda corpus: VECS)
    assert idiom.get_vec_of_word("most_frequent") is VECS


def test_get_vec_of_word_unknown():
    with pytest.raises(ValueError):
        idiom.get_vec_of_word("nonsense")


def test_high_dims_basics():
    assert hd.expected_dot_product_variance(10) == pytest.approx(0.1)
    v = hd.generate_random_unit_vectors(8, 6)
    assert v.shape == (6, 8)
    a = hd.measure_pairwise_angles(v)
    assert len(a) == 15 and ((a >= 0) & (a <= 180)).all()
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hd.nearly_orthogonal_vectors_johnson_lindenstrauss(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert hd.nearly_orthogonal_vectors_johnson_lindenstrauss(100, 0.1) >= 100
