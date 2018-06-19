# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:41:57 2017
Code to compare t-map with gene expression
Most codes are from alleninf (https://github.com/chrisfilo/alleninf)
@author: Zhipeng
"""

import pandas as pd
import numpy as np
import json
import urllib.request
import nibabel as nb
import numpy.linalg as npl
from scipy.stats.stats import pearsonr, spearmanr
import seaborn as sns
import os.path

# 1. Get Gene probe_id using API only works for one gene input


def get_probes_from_genes(gene_name):
    print('fetching probe ids for %s' % gene_name)
    api_url = "http://api.brain-map.org/api/v2/data/query.json"
    api_query = "?criteria=model::Probe"
    api_query += ",rma::criteria,[probe_type$eq'DNA']"
    api_query += ",products[abbreviation$eq'HumanMA']"
    api_query += ",gene[acronym$eq%s]" % gene_name
    api_query += ",rma::options[only$eq'probes.id','name']"
    response = urllib.request.urlopen(api_url + api_query)
    raw_json = response.read().decode("utf-8")
    data = json.loads(raw_json)
    d = {probe['id']: probe['name'] for probe in data['msg']}
    print('%s probes were found for %s:%s' % (len(d), gene_name, list(d.values())))
    if not d:
        d = 0
        print("Could not find any probes for %s gene.This could be a bug, as all the gene from my gene list should be found" % gene_name)
    return d

# 2. Get probe_expression_value, well_id, donor_names


def get_expression_values_from_probe_ids(probe_ids):
    print('fethcing expression value from probe_ids')
    if not isinstance(probe_ids, list):
        probe_ids = list(probe_ids)
    # in case there are white spaces in gene names
    probe_ids = ["'%s'" % probe_id for probe_id in probe_ids]
    api_url = "http://api.brain-map.org/api/v2/data/query.json"
    api_query = "?criteria=service::human_microarray_expression[probes$in%s]" % (
        ','.join(probe_ids))
    response = urllib.request.urlopen(api_url + api_query)
    raw_json = response.read().decode("utf-8")
    data = json.loads(raw_json)

    expression_values = [[float(expression_value) for expression_value in data[
        "msg"]["probes"][i]["expression_level"]] for i in range(len(probe_ids))]
    well_ids = [sample["sample"]["well"] for sample in data["msg"]["samples"]]
    donor_names = [sample["donor"]["name"]
                   for sample in data["msg"]["samples"]]
    well_coordinates = [sample["sample"]["mri"]
                        for sample in data["msg"]["samples"]]
    return expression_values, well_ids, donor_names


# combined function
def get_expression_values_from_gene_list(gene_list):
    all_gene_value = {}
    n = 1
    s = f = 0
    for gene_name in gene_list:
        print('********* working on %s, %d of %d ***********' % (gene_name, n, len(gene_list)))
        probes_dict = get_probes_from_genes(gene_name)
        if probes_dict != 0:
            expression_values, well_ids, donor_names = get_expression_values_from_probe_ids(
                probes_dict.keys())
            combined_expression_values = combine_expression_values(expression_values)
            all_gene_value[gene_name] = combined_expression_values
            s += 1
        else:
            print('fail to get %s expression' % gene_name)
            f += 1
        n += 1
        print('success %d, fail %d' % (s, f))

    all_gene_expression = pd.DataFrame.from_dict(all_gene_value)
    all_gene_expression.to_csv('tmp_all_gene_expression.csv', index=0)
    return all_gene_expression, well_ids


# 3. Get MNI from csv file for well_id
def get_mni_coordinates_from_wells(well_ids, csvfile=r'C:\Users\Zhipeng\Desktop\AllenBrain papers\alleninf-master\alleninf\data\corrected_mni_coordinates.csv'):
    print('get MNI xyz for wells')
    frame = pd.read_csv(csvfile, header=0, index_col=0)
    return list(frame.ix[well_ids].itertuples(index=False))

# 4. Combine probe_expression_value


def combine_expression_values(expression_values, method="average"):
    print('averaging probe values')
    if method == "average":
        return list(np.array(expression_values).mean(axis=0))
    elif method == "pca":
        from sklearn.decomposition import TruncatedSVD
        pca = TruncatedSVD(n_components=1)
        pca.fit(np.array(expression_values))
        return list(pca.components_[0, :])
    else:
        raise Exception("Uknown method")

# 5.Get values at MNI using radius, and apply a mask


def get_values_at_locations(nifti_file, locations, radius, mask_file=None,  verbose=False):
    print('get nii values at well\'s location')
    values = []
    nii = nb.load(nifti_file)
    data = nii.get_data()

    if mask_file:
        mask_data = nb.load(mask_file).get_data()
        mask = np.logical_and(np.logical_not(np.isnan(mask_data)), mask_data > 0)
    else:
        if verbose:
            print("No mask provided - using implicit (not NaN, not zero) mask")
        mask = np.logical_and(np.logical_not(np.isnan(data)), data != 0)

    for location in locations:
        coord_data = [round(i) for i in nb.affines.apply_affine(npl.inv(nii.affine), location)]
        sph_mask = (np.zeros(mask.shape) == True)
        if radius:
            sph = tuple(get_sphere(coord_data, vox_dims=nii.header.get_zooms(),
                                   r=radius, dims=nii.shape).T)
            sph_mask[sph] = True
        else:
            # if radius is not set use a single point
            sph_mask[coord_data[0], coord_data[1], coord_data[2]] = True

        roi = np.logical_and(mask, sph_mask)

        # If the roi is outside of the statmap mask we should treat it as a missing value
        if np.any(roi):
            val = data[roi].mean()
        else:
            val = np.nan
        values.append(val)
    return values


def get_sphere(coords, r, vox_dims, dims):
    """ # Return all points within r mm of coordinates. Generates a cube
    and then discards all points outside sphere. Only returns values that
    fall within the dimensions of the image."""
    r = float(r)
    xx, yy, zz = [slice(-r / vox_dims[i], r / vox_dims[
                        i] + 0.01, 1) for i in range(len(coords))]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    sphere = cube[:, np.sum(np.dot(np.diag(
        vox_dims), cube) ** 2, 0) ** .5 <= r]
    sphere = np.round(sphere.T + coords)
    return sphere[(np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, dims), 1) <= -1), :].astype(int)


# 6. correlation figure:
def correlation_plot(data, labels):
    sns_plot1 = sns.jointplot(labels[0], labels[1], data, kind="reg")
    sns_plot2 = sns.jointplot(labels[0], labels[1], data, stat_func=spearmanr, kind="hex")
    sns_plot1.savefig('pearson.tiff', dpi=800)
    sns_plot2.savefig('spearman.tiff', dpi=800)


# 8 random chose some gene
def random_gene(sample_n, exclude_genes, gene_list_csv=r'C:\Users\Zhipeng\Desktop\AllenBrain papers\alleninf-master\alleninf\data\gene_list.csv'):
    frame = pd.read_csv(gene_list_csv, header=0)
    # frame=frame.drop_duplicates()
    # frame.to_csv(csvfile,index=0)
    gene_name = list(frame['Gene_name'])
    clean_gene_name = [n for n in gene_name if n not in exclude_genes]
    random_gene = [str(n) for n in list(np.random.choice(clean_gene_name, sample_n))]
    return random_gene

# get nii files


def find_nii(input_path):
    all_file = []
    file_names = []
    for file in os.listdir(input_path):
        if file.endswith(".nii"):
            tmp = os.path.join(input_path, file)
            all_file.append(tmp)
    for n in all_file:
        [stat_name, ext] = os.path.basename(n).split('.')
        file_names.append(stat_name)
    return all_file, file_names


def boostrap_nii_vs_gene_list(stat_map, well_ids, gene_expression_table, boot_n=5000):
    # if boot_n=0, no bootstrap will be performed, raw data table will be the output
    mni_coordinates = get_mni_coordinates_from_wells(well_ids)
    nifti_values = get_values_at_locations(
        stat_map, mni_coordinates, mask_file=[], radius=3, verbose=True)
    [stat_name, ext] = os.path.basename(stat_map).split('.')
# add as 1st col in table
    tmp_table = gene_expression_table.copy()  # get new all_null from bak
    tmp_table.insert(loc=0, column=stat_name, value=nifti_values, allow_duplicates=True)
    tmp_table.dropna(axis=0, inplace=True)

# bootstrap the table
    if boot_n > 0:
        all_r_values = []
# bootstrap resampling is done on whole table, spearman is then performed on each
# column against 1st column.
        for boot_n in range(boot_n):
            print('bootstraping for %s: %d' % (stat_name, boot_n))
            h = tmp_table.shape[0]
            idx = np.random.choice(h, h) # Here it should use permutation! Need to change----need to fix
            resampled_table = tmp_table.iloc[idx, :] # only shuffle the 1: columns----need to fix
            tmp_r = spearman_corrwith(resampled_table)
            all_r_values.append(tmp_r)
        return all_r_values, tmp_table
    else:
        return tmp_table


def spearman_corrwith(data):
    # correlate with 1st column of the table
    col1 = data.iloc[:, 0]
    r_values = []
    for n in list(range(1, len(data.columns))):
        r, _ = spearmanr(col1, data.iloc[:, n])
        r_values.append(r)
    return r_values

# not very useful functions
# def fixed_effects(data, labels):
#     S_corcoeff, S_p_val = spearmanr(data[labels[0]], data[labels[1]])
#     p_corcoeff, P_p_val = pearsonr(data[labels[0]], data[labels[1]])
#     sns_plot1=sns.jointplot(labels[0], labels[1], data, kind="reg")
#     sns_plot2=sns.jointplot(labels[0], labels[1], data, stat_func=spearmanr,kind="hex")
#     sns_plot1.savefig('pearson.tiff',dpi=800)
#     sns_plot2.savefig('spearman.tiff',dpi=800)
#     return p_corcoeff, P_p_val, S_corcoeff, S_p_val, nii_value, exp_value
#
#
#
# def excute_simple_alleninf (stat_map,gene_name):
#     probes_dict = get_probes_from_genes(gene_name)
#     expression_values, well_ids, donor_names = get_expression_values_from_probe_ids(probes_dict.keys())
#     combined_expression_values=combine_expression_values(expression_values)
#     mni_coordinates = get_mni_coordinates_from_wells(well_ids)
#     nifti_values = get_values_at_locations(stat_map, mni_coordinates, mask_file=[], radius=3, verbose=True)
#
#     names = ["NIFTI values", "%s expression" % gene_name, "donor ID"]
#     data = pd.DataFrame(np.array(
#         [nifti_values, combined_expression_values, donor_names]).T, columns=names)
#     data = data.convert_objects(convert_numeric=True)
#     #data=data.apply(pd.to_numeric,errors='ignore')
#     data.dropna(axis=0, inplace=True)
#
#     P_corcoeff, P_p_val, S_corcoeff, S_p_val,nii_value,exp_value=fixed_effects(data, ["NIFTI values", "%s expression" % gene_name])
#     results={'pearson':[P_corcoeff,P_p_val],'spearman':[S_corcoeff,S_p_val]}
#     results_table=pd.DataFrame.from_dict(results).T
#     #correlation_plot(data, ["NIFTI values", "%s expression" % gene_name])
#     return results_table, nii_value, exp_value
#
#
# def get_values(stat_map,gene_name):
#     probes_dict = get_probes_from_genes(gene_name)
#     expression_values, well_ids, donor_names = get_expression_values_from_probe_ids(probes_dict.keys())
#     combined_expression_values=combine_expression_values(expression_values)
#     mni_coordinates = get_mni_coordinates_from_wells(well_ids)
#     nifti_values = get_values_at_locations(stat_map, mni_coordinates, mask_file=[], radius=3, verbose=True)
#
#     [stat_name,ext]=os.path.basename(stat_map).split('.')
#     # stat_name=os.path.splitext(os.path.basename(stat_map))[0]
#     names = ["%s" % stat_name ,"%s" % gene_name]
#     data = pd.DataFrame(np.array(
#         [nifti_values, combined_expression_values]).T, columns=names)
#     #data = data.convert_objects(convert_numeric=True)
#     data=data.apply(pd.to_numeric,errors='coerce')
#     data.dropna(axis=0,inplace=True)
    # return data
