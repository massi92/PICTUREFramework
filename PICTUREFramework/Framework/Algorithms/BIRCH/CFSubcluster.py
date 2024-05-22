import numpy as np
from math import sqrt

class _CFSubcluster:  # QUESETO CORRISPONDE A MICROCLUSTER
    """Each subcluster in a CFNode is called a CFSubcluster.

    A CFSubcluster can have a CFNode has its child.

    Parameters
    ----------
    linear_sum : ndarray of shape (n_features,), default=None
        Sample. This is kept optional to allow initialization of empty
        subclusters.

    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.

    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    squared_sum_ : float
        Sum of the squared l2 norms of all samples belonging to a subcluster.

    centroid_ : ndarray of shape (branching_factor + 1, n_features)
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``CFNode.centroids_`` is called.

    child_ : _CFNode
        Child Node of the subcluster. Once a given _CFNode is set as the child
        of the _CFNode, it is set to ``self.child_``.

    sq_norm_ : ndarray of shape (branching_factor + 1,)
        Squared norm of the subcluster. Used to prevent recomputing when
        pairwise minimum distances are computed.
    """

    def __init__(self, *, linear_sum=None):
        if linear_sum is None:
            self.n_samples_ = 0
            self.squared_sum_ = 0.0
            self.centroid_ = self.linear_sum_ = 0
        else:
            self.n_samples_ = 1
            self.centroid_ = self.linear_sum_ = linear_sum
            self.squared_sum_ = self.sq_norm_ = np.dot(
                self.linear_sum_, self.linear_sum_
            )
        self.child_ = None

    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.squared_sum_ += subcluster.squared_sum_
        self.centroid_ = self.linear_sum_ / self.n_samples_
        self.sq_norm_ = np.dot(self.centroid_, self.centroid_)

    def merge_subcluster(self, nominee_cluster, threshold):
        """Check if a cluster is worthy enough to be merged. If
        yes then merge.
        """
        new_ss = self.squared_sum_ + nominee_cluster.squared_sum_
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        new_centroid = (1 / new_n) * new_ls
        new_sq_norm = np.dot(new_centroid, new_centroid)

        # The squared radius of the cluster is defined:
        #   r^2  = sum_i ||x_i - c||^2 / n
        # with x_i the n points assigned to the cluster and c its centroid:
        #   c = sum_i x_i / n
        # This can be expanded to:
        #   r^2 = sum_i ||x_i||^2 / n - 2 < sum_i x_i / n, c> + n ||c||^2 / n
        # and therefore simplifies to:
        #   r^2 = sum_i ||x_i||^2 / n - ||c||^2
        sq_radius = new_ss / new_n - new_sq_norm

        if sq_radius <= threshold ** 2:
            (
                self.n_samples_,
                self.linear_sum_,
                self.squared_sum_,
                self.centroid_,
                self.sq_norm_,
            ) = (new_n, new_ls, new_ss, new_centroid, new_sq_norm)
            return True
        return False

    @property
    def radius(self):
        """Return radius of the subcluster"""
        # Because of numerical issues, this could become negative
        sq_radius = self.squared_sum_ / self.n_samples_ - self.sq_norm_
        return sqrt(max(0, sq_radius))