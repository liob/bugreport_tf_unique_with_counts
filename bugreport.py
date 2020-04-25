#!/usr/bin/env python
# -*- coding: utf-8 -*-

from timeit import timeit, repeat
import torch
import tensorflow as tf
from pandas import DataFrame
import numpy as np
from tabulate import tabulate


def run_experiment(function, data, framework, dtype, n_runs=10, n_repeats=5, **kwargs):
    global df
    def _func(data, kwargs):
        return lambda: function(data, **kwargs)
    r = repeat(_func(data, kwargs), repeat=n_repeats ,number=n_runs)
    vals = {'framework': framework,
            'mean (ms)': np.mean(r)*1000,
            'std (ms)':  np.std(r)*1000,
            'dtype':     dtype}
    df = df.append(vals, ignore_index=True)


df = DataFrame()

# generate sample data
a = np.random.randint(1024, size=1000000)
tf_int32 = tf.convert_to_tensor(a, dtype=tf.int32)
tf_int16 = tf.convert_to_tensor(a, dtype=tf.int16)
tf_float32 = tf.convert_to_tensor(a, dtype=tf.float32)
tf_float16 = tf.convert_to_tensor(a, dtype=tf.float16)
pt_int32 = torch.from_numpy(a.astype(np.int32)).cuda()
pt_int16 = torch.from_numpy(a.astype(np.int16)).cuda()
pt_float32 = torch.from_numpy(a.astype(np.float32)).cuda()
pt_float16 = torch.from_numpy(a.astype(np.float16)).cuda()


# run tf experiments
run_experiment(tf.unique_with_counts, tf_int32,   'tensorflow', 'int32')
run_experiment(tf.unique_with_counts, tf_int16,   'tensorflow', 'int16')
run_experiment(tf.unique_with_counts, tf_float32, 'tensorflow', 'float32')
run_experiment(tf.unique_with_counts, tf_float16, 'tensorflow', 'float16')

# run pytorch experiments
run_experiment(torch.unique, pt_int32,   'pytorch', 'int32',   return_counts=True)
run_experiment(torch.unique, pt_int16,   'pytorch', 'int16',   return_counts=True)
run_experiment(torch.unique, pt_float32, 'pytorch', 'float32', return_counts=True)
run_experiment(torch.unique, pt_float16, 'pytorch', 'float16', return_counts=True)

#print(df)

hdr = ['framework', 'mean (ms)', 'std (ms)', 'dtype']
print(tabulate(df[hdr], headers=hdr))