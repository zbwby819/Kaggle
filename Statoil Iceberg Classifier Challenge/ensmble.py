# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:56:34 2018

@author: Administrator
"""

import os
import numpy as np 
import pandas as pd 
from subprocess import check_output
import seaborn as sns

sub_path = "sub"
all_files = os.listdir(sub_path)

outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
concat_sub.head()

concat_sub.corr()
g = sns.heatmap(concat_sub[["is_iceberg_0","is_iceberg_1","is_iceberg_2","is_iceberg_3","is_iceberg_4","is_iceberg_5","is_iceberg_6","is_iceberg_7","is_iceberg_8"]].corr(),
                            annot=True, fmt = ".2f", cmap = "coolwarm")

max = concat_sub.iloc[:, 1:10].max(axis=1)
min = concat_sub.iloc[:, 1:10].min(axis=1)
mean = concat_sub.iloc[:, 1:10].mean(axis=1)
mean2 = concat_sub.iloc[:, 2:9].mean(axis=1)


concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:10].max(axis=1)
concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:10].min(axis=1)
concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:10].mean(axis=1)
concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:5].median(axis=1)

cutoff_lo = 0.86
cutoff_hi = 0.15

concat_sub['is_iceberg'] = concat_sub['is_iceberg_mean']

concat_sub['is_iceberg'] = np.where(np.all(concat_sub['is_iceberg_max'].values > cutoff_lo), 1, 
                                    np.where(np.all(concat_sub['is_iceberg_min'].values < cutoff_hi),0, concat_sub['is_iceberg_base']))
concat_sub['is_iceberg'] = np.clip(concat_sub['is_iceberg'].values, 0.001, 0.999)



concat_sub[['id', 'is_iceberg']].to_csv('mean.csv', index=False, float_format='%.6f')

concat_sub['is_iceberg'] = concat_sub['is_iceberg_mean']
concat_sub[['id', 'is_iceberg']].to_csv('stack_mean.csv', 
                                        index=False, float_format='%.6f')

sub_base = pd.read_csv('../input/submission38-lb01448/submission43.csv')
concat_sub['is_iceberg_base'] = sub_base['is_iceberg']
concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1), concat_sub['is_iceberg_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1),
                                             concat_sub['is_iceberg_min'], 
                                             concat_sub['is_iceberg_base']))

concat_sub[['id', 'is_iceberg']].to_csv('submission54.csv', 
                                        index=False, float_format='%.6f')




concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:,1:6] < cutoff_hi, axis=1),
                                             concat_sub['is_iceberg_min'], 
                                    np.where(np.all(concat_sub.iloc[:,1:6] > cutoff_lo, axis=1),concat_sub['is_iceberg_max'],
                                             concat_sub['is_iceberg_base']))
