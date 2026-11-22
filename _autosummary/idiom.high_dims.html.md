# idiom.high_dims

Tools to study high-dimensional statistics.

### Functions

| [`convert_angle`](#idiom.high_dims.convert_angle)(angle, \*[, input_unit, ...])        | Converts an angle between different units: degrees, radians, and cosine.                                                                                                                                                                                   |
|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`convert_epsilon_angle`](#idiom.high_dims.convert_epsilon_angle)(angle, \*[, ...])            | Computes the absolute difference between a straight angle (180 degrees or π radians) and the input angle, converting it to the desired unit.                                                                                                               |
| [`expected_dot_product_variance`](#idiom.high_dims.expected_dot_product_variance)(n)                   | Computes the expected variance of the dot product between two random unit vectors in an n-dimensional space.                                                                                                                                               |
| [`generate_random_unit_vectors`](#idiom.high_dims.generate_random_unit_vectors)(dim[, num_vectors])   | Generates `num_vectors` random unit vectors in `dim`-dimensional space.                                                                                                                                                                                    |
| [`measure_pairwise_angles`](#idiom.high_dims.measure_pairwise_angles)(vectors)                   | Computes the pairwise angles (in degrees) between unit vectors.                                                                                                                                                                                            |
| [`nearly_orthogonal_vectors`](#idiom.high_dims.nearly_orthogonal_vectors)(n, epsilon[, ...])       | Estimates the min number of nearly orthogonal vectors that can fit in an n-dimensional.                                                                                                                                                                    |
| [`nearly_orthogonal_vectors_equiangular`](#idiom.high_dims.nearly_orthogonal_vectors_equiangular)(n, epsilon)  | Estimates a (provable) lower bound for the maximum number N(ε, n) of unit vectors in an n-dimensional Euclidean space that can be chosen so that every pair of distinct vectors has an inner product (cosine of the angle between them) bounded by ε, i.e. |
| [`nearly_orthogonal_vectors_johnson_lindenstrauss`](#idiom.high_dims.nearly_orthogonal_vectors_johnson_lindenstrauss)(n) | Estimates the number of nearly orthogonal vectors that can fit in an n-dimensional space given an orthogonality tolerance epsilon.                                                                                                                         |
| [`run_high_dim_experiment`](#idiom.high_dims.run_high_dim_experiment)([dims, n_points, ...])     | Runs HighDimStats across multiple dimensions and returns results as a DataFrame.                                                                                                                                                                           |

### Classes

| `HighDimStats`([data, dim, n_points, generator])   |    |
|----------------------------------------------------|----|

### idiom.high_dims.convert_angle(angle, , input_unit='degrees', output_unit='cosine')

Converts an angle between different units: degrees, radians, and cosine.

* **Parameters:**
  * **angle** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – The angle to convert.
  * **input_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'cosine'`, `'degrees'`, `'radians'`]) – The unit of the input angle (‘degrees’, ‘radians’, ‘cosine’).
  * **output_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'cosine'`, `'degrees'`, `'radians'`]) – The unit to convert to (‘degrees’, ‘radians’, ‘cosine’).
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  The converted angle.

```pycon
>>> float(convert_angle(90, input_unit='degrees', output_unit='radians'))
1.570796...
>>> float(convert_angle(0.5, input_unit='radians', output_unit='degrees'))
28.6478...
>>> float(convert_angle(0, input_unit='degrees', output_unit='cosine'))
1.0
```

### idiom.high_dims.convert_epsilon_angle(angle, , input_unit='degrees', output_unit='cosine')

Computes the absolute difference between a straight angle (180 degrees or π radians)
and the input angle, converting it to the desired unit.

* **Parameters:**
  * **angle** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – The input angle.
  * **input_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'cosine'`, `'degrees'`, `'radians'`]) – The unit of the input angle (‘degrees’, ‘radians’, ‘cosine’).
  * **output_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'cosine'`, `'degrees'`, `'radians'`]) – The unit to convert the epsilon angle to (‘degrees’, ‘radians’, ‘cosine’).
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  The converted epsilon angle.

```pycon
>>> float(convert_epsilon_angle(0, input_unit='degrees', output_unit='radians'))
1.5707...
```

### idiom.high_dims.expected_dot_product_variance(n)

Computes the expected variance of the dot product between two random unit vectors
in an n-dimensional space.

* **Parameters:**
  **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Dimension of the space
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  Expected variance of the dot product

```pycon
>>> expected_dot_product_variance(10)
0.1
>>> expected_dot_product_variance(100)
0.01
```

### idiom.high_dims.generate_random_unit_vectors(dim, num_vectors=100)

Generates `num_vectors` random unit vectors in `dim`-dimensional space.

* **Parameters:**
  * **dim** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Dimension of the space
  * **num_vectors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of vectors to generate
* **Return type:**
  `ndarray`
* **Returns:**
  A (num_vectors, dim) numpy array of unit vectors

```pycon
>>> vectors = generate_random_unit_vectors(5, 3)
>>> np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
True
```

### idiom.high_dims.measure_pairwise_angles(vectors)

Computes the pairwise angles (in degrees) between unit vectors.

* **Parameters:**
  **vectors** (`ndarray`) – A numpy array of shape (num_vectors, dim)
* **Return type:**
  `ndarray`
* **Returns:**
  A numpy array of pairwise angles in degrees

```pycon
>>> vecs = generate_random_unit_vectors(10, 5)
>>> angles = measure_pairwise_angles(vecs)
>>> bool((0 <= angles).all() and (angles <= 180).all())
True
```

### idiom.high_dims.nearly_orthogonal_vectors(n, epsilon, epsilon_unit='cosine')

Estimates the min number of nearly orthogonal vectors that can fit in an
n-dimensional.

### idiom.high_dims.nearly_orthogonal_vectors_equiangular(n, epsilon, epsilon_unit='cosine')

Estimates a (provable) lower bound for the maximum number N(ε, n) of unit vectors in an n-dimensional
Euclidean space that can be chosen so that every pair of distinct vectors has an inner product (cosine
of the angle between them) bounded by ε, i.e.

> ```
> |v_i · v_j|
> ```

>  <= epsilon,   for all i ≠ j.

Special cases:

> - If epsilon = 0, the vectors must be exactly orthogonal, so the maximum is exactly n.
> - If epsilon >= 1, every pair of unit vectors trivially satisfies 

>   ```
>   |v_i · v_j|
>   ```

>    ≤ 1, so one may
>   pack infinitely many such vectors.

For 0 < epsilon < 1, one standard (Johnson–Lindenstrauss) argument shows one can pack roughly
: exp((epsilon\*\*2 \* n) / 8)

vectors. However, more refined techniques from spherical coding theory and the Kabatjanskii–Levenstein
bound yield a stronger lower bound of the form

> N(ε, n) >= exp((epsilon\*\*2 \* n) / 2).

For example, with n ≈ 1500 and epsilon ≈ 0.1 the refined bound gives
: N(0.1, 1500) >= exp((0.1\*\*2 \* 1500) / 2) = exp(7.5) ≈ 1800.

This function returns the larger of the trivial bound (n) and the refined bound exp((epsilon² \* n) / 2).

For more details on these results and their derivations, see, for example,
“Equiangular lines” on Wikipedia:

```default
https://en.wikipedia.org/wiki/Equiangular_lines
```

#### NOTE
This is an asymptotic (existential) lower bound and should be considered a heuristic estimate.

* **Parameters:**
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Dimension of the Euclidean space (n >= 1).
  * **epsilon** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Tolerance for “near-orthogonality” (0 <= epsilon < 1).
  * **epsilon_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'cosine'`, `'degrees'`, `'radians'`]) – The unit of epsilon (‘cosine’, ‘degrees’, or ‘radians’).
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  An estimate for the maximum number of nearly orthogonal unit vectors.

# >>> nearly_orthogonal_vectors_equiangular(100, 0.1)
# 100
# >>> nearly_orthogonal_vectors_equiangular(1000, 0.1)
# 1000
# >>> nearly_orthogonal_vectors_equiangular(10000, 0.1)
# 5184705528587109793792

# With different epsilon units:

# >>> nearly_orthogonal_vectors_equiangular(100000, 0, epsilon_unit=’degrees’)
# 100000
# >>> nearly_orthogonal_vectors_equiangular(1000, 5, epsilon_unit=’degrees’)

### idiom.high_dims.nearly_orthogonal_vectors_johnson_lindenstrauss(n, epsilon=0.01, epsilon_unit='cosine')

Estimates the number of nearly orthogonal vectors that can fit in an n-dimensional
space given an orthogonality tolerance epsilon.

See [https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma#Statement](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma#Statement)

* **Parameters:**
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Dimension of the space
  * **epsilon** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Tolerance level for near-orthogonality
  * **epsilon_unit** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'cosine'`, `'degrees'`, `'radians'`]) – The unit of epsilon (‘cosine’, ‘degrees’, or ‘radians’).
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  Estimated number of nearly orthogonal vectors

# >>> nearly_orthogonal_vectors_johnson_lindenstrauss(100, 0.1)
# 100
# >>> nearly_orthogonal_vectors_johnson_lindenstrauss(1000, 0.1)
# 1484
# >>> nearly_orthogonal_vectors_johnson_lindenstrauss(10000, 0.1)
# 51847055285871095840768

### idiom.high_dims.run_high_dim_experiment(dims=(2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000), n_points=5000, generator=<bound method RandomState.uniform of RandomState(MT19937)>)

Runs HighDimStats across multiple dimensions and returns results as a DataFrame.

* **Parameters:**
  * **dims** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) – List of dimensions to evaluate.
  * **n_points** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Number of points per dimension.
  * **generator** (*callable*) – Random generator function (default: uniform).
* **Returns:**
  Contains statistics for each dimension.
* **Return type:**
  pd.DataFrame
