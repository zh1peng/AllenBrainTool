### Allen Brain Atlas tool

#### Need a new documentation and webpage for this.



1. get gene expression from Allen Brain API (http://api.brain-map.org/api/v2/data/query.json)


2. average or pca gene expression using muliple
3. corrected mni for each well-id
4. get t-value from mni (average with r)
5. random gene selection
6. bootstrap on spearman correlation, which allow to build a null model to test if the target gene is 'real' correlate with the t-map.
7. find nii files

1-4 features are from alleninf (https://github.com/chrisfilo/alleninf), and changed to python 3x. 



### Before use

1.change line 173 to the gene list (i.e. gene_list_csv) available from allen brain atlas. This file is converted from official pdf documentation. Some genes are not available when using the toolbox to retreive. 

2.change line 91 (csvfile) to the corrected_mni_coordinates.csv. This csv file is from https://github.com/chrisfilo/alleninf. 



#### Example1 Get some random gene and run bootstrap

```python
import pandas as pd
import os
from zhipeng_alleninf import *

all_file, file_names = find_nii(
    r"-----------------nii file dir-------------------")

gene_list = random_gene(3, 'DRD1')
all_gene_expression, well_ids = get_expression_values_from_gene_list(gene_list)
stat_map = all_file[0]
test, data_table = boostrap_nii_vs_gene_list(stat_map, well_ids, all_gene_expression, boot_n=500)
# test is [r1 r2 r3 r4..rn]*bootstrap_n results.
# rn is spearman r for nth gene against 1st column which is the nii values.
# use np.percentile to find percentile
```

#### Example2 Get gene of interest and get data table

```python
import pandas as pd
import os
from zhipeng_alleninf import *

all_file, file_names = find_nii(
    r"--------------------nii file dir-------------------")
gene_list = ['DRD1', 'DRD2', 'DRD3']

all_gene_expression, well_ids = get_expression_values_from_gene_list(gene_list)

stat_map = all_file[0] #select the 1st stat map

data_table = boostrap_nii_vs_gene_list(stat_map, well_ids, all_gene_expression, boot_n=0)
#if boot_strap set to 0, only return data_table

# how to plot
import pylab as plt
import seaborn as sns
from scipy.stats.stats import pearsonr, spearmanr
labels = list(data_table.keys())
sns_plot1 = sns.jointplot(labels[0], labels[1], data_table.iloc[:, [0, 1]], kind="reg")
sns_plot2 = sns.jointplot(labels[0], labels[1], data_table.iloc[:,
                                                                [0, 1]], stat_func=spearmanr, kind="reg")
```

#### 

