from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from numpy import array
import numpy.linalg as la
from collections import defaultdict
import copy

def refill_norm_data(input_data, copy_data=False, prec=10e-10, max_iter=100, silent=False, return_groups=False):
    """
    Fill missing data for normally distributed numpy.array using the EM Algorithm

    :param input_data: 2D numpy.ndarray or Pandas DataFrame with missing values.
    :param copy_data: if True will return new data Matrix. Otherwise will update input_data matrix. if input_data is a
    DataFrame, this is automatically set to False and the DF will be updated.
    :param prec: stop after squared change in mean becomes less than x
    :param max_iter: maximum number of iterations of EM algo
    :param silent: if False (default) will print summary of run
    :return:
    """

    # Find all combinations of missing columns and their locations

    if isinstance(input_data, pd.core.frame.DataFrame):
        copy_data = False
        data = input_data.values

    elif copy_data is False:
        data = input_data

    else:
        data = copy.copy(input_data)

    if silent is False:
        miss_count = np.count_nonzero(np.isnan(data))

    position_dict = defaultdict(list)

    for i, cc in enumerate(data):
        pos = np.where(np.isnan(cc))

        if pos[0].shape[0]:
            name = str(pos)
        else:
            name = 'None'

        position_dict[name] += [i]

    # Remove the group of data points that aren't missing any observations
    # We use them as prior below
    missing_keys = position_dict.keys()
    missing_keys.remove("None")
    n_cols = data.shape[1]

    # Instantiate MissingColumnGroup objects
    # this makes the bookkeeping much easier.
    missing_groups = []
    all_missing = []

    for key in missing_keys:
        mdg = MissingColumnGroup(position_dict[key],
                                 eval(key)[0],
                                 n_cols)

        all_missing += position_dict[key]
        missing_groups.append(mdg)

    all_missing.sort()

    # use fully observed data moments as prior
    data0 = data[position_dict["None"]]
    mean0 = data0.mean(axis=0)
    cov0 = np.cov(data0.T)

    # Initialize with Posterior Moments = Prior Moments

    cov = cov0
    mean = mean0

    # Run EM Algorithm

    # setup

    n = data.shape[0]  # number of observations
    n0 = data0.shape[0]  # number of observations without missing data
    n1 = n - n0  # number of observations with missing data
    n0m0m0T = n0 * np.outer(mean0, mean0)  # constant that's reused below.

    i = 0
    while True:

        # Expectation Step

        # find expected value of missing data points.
        for mdg in missing_groups:
            mdg.update_group_stats(cov, mean)
            mdg.update_missing_data(data)

        # Maximization Step

        # find MAP estimates of posterior mean and covariance
        data1 = data[all_missing]
        last_mean = mean
        # de-centralized scatter matrix
        S = data1.T.dot(data1)
        # mean of data with missing observations
        mean1 = data1.mean(axis=0)
        # posterior mean:
        # weighted average of prior and missing data means
        mean = (n0 * mean0 + n1 * mean1) / n
        # posterior covariance:
        # new covariance is the weighted average of prior and missing data covariances
        # plus an adjustment for uncertainty in the mean
        cov = (n0 * cov0 + S + n0m0m0T - n * np.outer(mean, mean)) / n

        # check precision
        dif2 = (mean - last_mean).dot(mean - last_mean)
        i += 1

        if np.isnan(dif2):
            print("No Missing Data")
            return

        if dif2 < prec or i > max_iter:
            break

    if silent is False:
        total_obs = np.asarray(data.shape).prod()
        print('Filled {} missing observations out of {}'.format(miss_count, total_obs))
        print('{} rows out of {} were missing data'.format(n1, n))
        print('Iterations : {}'.format(i))
        print('Squared difference of last 2 posterior mean iterations: {}'.format(dif2))

    if copy_data is True:
        return data

    if return_groups is True:
        return missing_groups


class MissingColumnGroup:

    def __init__(self, miss_pos, miss_col, n_col):
        """
        Book keeping object for finding expected value of all data points missing the same columns in
        norm_refill.refill_norm_data()

        :param miss_pos: row number of incomplete data
        :param miss_col: 1D np.array of which columns are missing
        :param n_col: data.shape[1]
        :return:
        """

        self.miss_pos = np.asarray(miss_pos)
        self.miss_col = miss_col
        self.there_col = np.setdiff1d(np.arange(n_col), miss_col)

        self.n_miss = miss_col.shape[0]
        self.n_there = self.there_col.shape[0]

        self.mean_m = np.empty(self.n_miss)
        self.mean_t = np.empty(self.n_there)

        self.cov_t = np.empty((self.n_there, self.n_there))
        self.cov_m_t = np.empty((self.n_miss, self.n_there))

        self.cov_inv_t = np.empty((self.n_there, self.n_there))

    def update_group_stats(self, cov, mean):
        """
        update column group sufficient stats from total data posterior
        :param cov: posterior covariance matrix
        :param mean: posterior mean
        :return: None
        """

        there = self.there_col
        miss = self.miss_col

        for i0, t0 in enumerate(there):
            for i1, t1 in enumerate(there):
                self.cov_t[i0, i1] = cov[t0, t1]

        for i0, m0 in enumerate(miss):
            for i1, t1 in enumerate(there):
                self.cov_m_t[i0, i1] = cov[m0, t1]

        self.cov_inv_t = la.inv(self.cov_t)

        self.mean_m = mean[self.miss_col]
        self.mean_t = mean[self.there_col]

    def update_missing_data(self, data):
        """
        :param data: data matrix
        :return:
        """

        pos = self.miss_pos
        there = self.there_col
        miss = self.miss_col
        cov_m_t = self.cov_m_t
        cov_inv_t = self.cov_inv_t
        mean_t = self.mean_t
        mean_m = self.mean_m

        data_t = data[pos][:, there]

        missing_values = mean_m + cov_m_t.dot(cov_inv_t).dot((data_t - mean_t).T).T

        # Numpy can't broadcast fill here because the index lists are different shapes
        for i, m in enumerate(miss):
            data[pos, m] = missing_values[:, i]
