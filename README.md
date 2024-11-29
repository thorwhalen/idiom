# idiom

Access and operations with word2vec data

To install:	```pip install idiom```

## Overview

The `idiom` package provides access to word vector data and useful functions to manipulate and analyze it. It includes functionalities for finding the closest words to a given word, calculating word frequencies, and working with various word vector models.

## Features

- **Closest Words**: Find the closest words to a given word based on cosine similarity.
- **Word Frequencies**: Access and manipulate word frequency data.
- **Word Vector Models**: Work with pre-trained word vector models such as FastText.
- **IDF Calculations**: Compute different types of Inverse Document Frequency (IDF) values.

## Usage

### Finding Closest Words

You can find the closest words to a given word using the `closest_words` function:

```python
from idiom import closest_words

# Example: Find the closest words to 'mad' that start with 'l'
starts_with_L = lambda x: x.startswith('l')
print(closest_words('mad', k=10, search_words=starts_with_L))
```

### Accessing Word Frequencies

You can access the most frequent words using the `most_frequent_words` function:

```python
from idiom import most_frequent_words

# Get the top 100,000 most frequent words
frequent_words = most_frequent_words(max_n_words=100000)
print(frequent_words)
```

### Working with Word Vectors

You can load and work with pre-trained word vectors using the `WordVec` class:

```python
from idiom import WordVec

# Initialize WordVec with default word vectors
word_vec = WordVec()

# Calculate the distance between two queries
distance = word_vec.dist('france capital', 'paris')
print(distance)
```

### IDF Calculations

You can compute different types of IDF values using the `_IDF` class:

```python
from idiom import idf

# Access logarithmic IDF values
log_idf = idf.logarithmic
print(log_idf)
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue on GitHub.

## License

This project is licensed under the MIT License.


