##  Measures of Similarity/Dissimilarity
A measure of similarity $ s: \mathcal{X} × \mathcal{X} → \mathbb{R} $ or of dissimilarity is introduced.
One type of dissimilarity measure is the distance function $ d: \mathcal{X} × \mathcal{X} → \mathbb{R}_{\geq 0} $, which must satisfy the following properties:
1. **Symmetry**: $ d(x, y) = d(y, x) $
2. **Non-Negativity**: $ d(x, x) ≥ 0 $
3. **Reflexivity**: $ d(x, y) = 0 $ if and only if $ x = y $
4. **Triangle inequality**: $ d(x, y) ≤ d(x, z) + d(z, y) $

Dissimilarity measures can vary based on the properties they satisfy:
- ***Pseudo-distance***: satisfies all properties of a distance function except reflexivity.
- ***Meta-distance***: satisfies all properties of a distance function except reflexivity and symmetry.
- ***Semi-metric***: satisfies all properties of a distance function except symmetry.

It is more convenient to operate with the normalised similarity measure $ s: \mathcal{X} × \mathcal{X} → [0, 1] $, which is defined as:
$$ s(x, y) = 1 - d(x, y) $$

### Comparing the Objects Having Quantitative Features
The most popular measures of dissimilarity to compare objects with quantitative features are:
1. **Minkowski distance**: which is defined as:
$$ ∀(x,y) \in \mathbb{R^2} \quad d(x, y) = (∑_{i=1}^{n} |x_i - y_i|^p)^{1/p}, \quad p \in [1,  ∞[ $$
   - If $ p = 1 $, it is *Manhattan distance*:
   $$ d(x, y) = ∑_{i=1}^{n} |x_i - y_i| $$
   - if $ p = 2 $, it is *Euclidean distance*:
   $$ d(x, y) = \sqrt{∑_{i=1}^{n} (x_i - y_i)^2} $$
   - If $ p = ∞ $, it is *Chebyshev distance*:
   $$ d(x, y) = \max(|x_i - y_i|) $$

2. **Mahalanobis Distance**: is a measure of the distance between a point and a distribution. 
It is a multi-dimensional generalization of the idea of measuring how many standard deviations away a point is from the mean of a distribution.
It is defined as:
$$ ∀(x,y) \in \mathbb{R^2} \quad d(x, y) = \sqrt{(x - y)^T S^{-1} (x - y)} $$
where $ S $ is the covariance matrix of the data.
The covariance matrix is a square matrix giving the covariance between each pair of elements of a random vector. The diagonal elements of the covariance matrix are the variances of each element of the vector, it's defined as:
$$ S = \frac{1}{n} ∑_{i=1}^{n} (x_i - \bar{x})(x_i - \bar{x})^T \quad \text{avec} \quad  \bar{x} = \frac{1}{n} ∑_{i=1}^{n} x_i $$
3.  **Bregman Divergence** is a measure of dissimilarity between two points in a space. It is defined as:
$$ D(x, y) = f(x) - f(y) - ∇f(y)^T (x - y) $$
where $ f(x) $ is a convex function and $ ∇f(x) $ is the gradient of $ f(x) $.
4. **Cosine Distance** is a measure of similarity between two non-zero vectors of an inner product space. It is defined as:
$$ d(x, y) = 1 - \frac{x^T y}{||x||_2 ||y||_2} $$
5. **Power Distance** is a measure of dissimilarity between two points in a space. It is defined as:
$$ d(x, y) = 1 - \frac{(x^T y)^p}{||x||_2^p ||y||_2^p} $$