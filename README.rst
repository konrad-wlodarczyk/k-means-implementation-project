.. These badges assume GitHub repository name and project name alignment.
.. Update <USER> if needed.

.. image:: https://api.cirrus-ci.com/github/konrad-wlodarczyk/k-means-implementation-project.svg?branch=main
    :alt: Build Status
    :target: https://cirrus-ci.com/github/konrad-wlodarczyk/k-means-implementation-project

.. image:: https://readthedocs.org/projects/k-means-implementation-project/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://k-means-implementation-project.readthedocs.io/en/latest/

.. image:: https://img.shields.io/coveralls/github/konrad-wlodarczyk/k-means-implementation-project/main.svg
    :alt: Coveralls
    :target: https://coveralls.io/github/konrad-wlodarczyk/k-means-implementation-project

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==============================
K-Means Implementation Project
==============================

A Python implementation of the **k-means clustering algorithm** built from scratch for educational and experimental purposes.

The goal of this project is to provide a clear, readable, and testable reference implementation of k-means without relying on high-level machine learning frameworks.

Overview
========

K-means is an unsupervised learning algorithm that partitions a dataset into *k* clusters by minimizing within-cluster variance. The algorithm proceeds iteratively:

1. Initialize *k* centroids
2. Assign each data point to the nearest centroid
3. Recompute centroids as the mean of assigned points
4. Repeat until convergence or a maximum number of iterations is reached

This implementation focuses on algorithmic transparency and correctness.

Features
========

- Pure Python implementation
- Configurable number of clusters
- Iterative centroid refinement
- Convergence based on centroid stability
- Prediction support for unseen data
- Unit-tested core logic

Project Structure
================

::

    k-means-implementation-project/
    ├── src/
    │   └── kmeans/
    │       ├── __init__.py
    │       └── kmeans.py
    ├── tests/
    │   └── test_kmeans.py
    ├── docs/
    ├── README.rst
    ├── pyproject.toml
    ├── tox.ini
    └── requirements.txt

Installation
============

Clone the repository:

::

    git clone https://github.com/konrad-wlodarczyk/k-means-implementation-project.git
    cd k-means-implementation-project

Create a virtual environment and install dependencies:

::

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Usage
=====

Basic example:

::

    from kmeans.kmeans import KMeans

    X = [[1, 2], [1, 4], [5, 8], [8, 8]]

    model = KMeans(n_clusters=2, max_iter=100)
    model.fit(X)

    labels = model.labels_
    centroids = model.centroids_

Testing
=======

Run the full test suite using:

::

    tox

or directly with:

::

    pytest

Documentation
=============

Documentation is generated using **Sphinx** and **tox**.

To build the documentation locally:

::

    tox -e docs

The generated HTML documentation will be available in:

::

    docs/_build/html

Contributing
============

Contributions are welcome. Please ensure that:

- Code is well-documented
- Tests are added or updated as necessary
- All tox environments pass before submitting a pull request

License
=======

This project is licensed under the MIT License.  
See the ``LICENSE`` file for details.

References
==========

- MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations*
- https://en.wikipedia.org/wiki/K-means_clustering

.. _pyscaffold-notes:

Note
====

This project has been set up using **PyScaffold 4.6**.  
For details and usage information see https://pyscaffold.org/.