import math
import pandas as pd
import numpy as np
import streamlit as st

from numpy.linalg import svd
from numpy import eye, asarray, dot, sum, diag
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from factor_analyzer.rotator import Rotator

def custom_scale(matrix):
    # Separate positive and negative values
    pos_vals = matrix[matrix > 0]
    neg_vals = matrix[matrix < 0]
    
    # Scale positive values from 0 to 1
    if pos_vals.size > 0:  # Check if there are any positive values
        pos_min = pos_vals.min()
        pos_max = pos_vals.max()
        matrix[matrix > 0] = (pos_vals - pos_min) / (pos_max - pos_min)
    
    # Scale negative values from -1 to 0
    if neg_vals.size > 0:  # Check if there are any negative values
        neg_min = neg_vals.min()
        neg_max = neg_vals.max()
        matrix[matrix < 0] = -1 + (neg_vals - neg_min) / (neg_max - neg_min)
    
    return matrix

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R)

def sort_rows_and_cutoff(rotated_loadings, construct_names, construct_definitions, threshold):
    # Eliminate values less than 0.9 (normalized between 0 to 1, adjust threshold as needed)
    rotated_loadings[np.abs(rotated_loadings) < threshold] = np.nan

    # Convert to DataFrame for easier manipulation, setting row names as the index
    df_loadings = pd.DataFrame(rotated_loadings, index=construct_names, columns=[i+1 for i in range(rotated_loadings.shape[1])])

    # Sort rows in descending order based on loadings, prioritizing earlier columns
    sorted_index = np.lexsort([-np.abs(df_loadings[col]) for col in reversed(df_loadings.columns)])
    df_loadings.insert (0, "Definition", construct_definitions)
    df_sorted = df_loadings.iloc[sorted_index]

    # Reset index if you want the row names to become a regular column in the Excel file
    df_sorted.reset_index(inplace=True)
    df_sorted.replace(np.nan, '', inplace=True)
    return df_sorted

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

# @st.cache_resource
def dim_reduction_rotation(data_type, algo, rotation_method, prefix, dim, eigen):
    # load the data
    if data_type == 'Correlation Matrix':
        data = pd.read_csv(f'data/{prefix}_correlation_with_name.csv', index_col=[0,1])
    elif data_type == 'Similarity Matrix':
        data = pd.read_csv(f'data/{prefix}_similarity_with_name.csv', index_col=[0,1])
    else:
        raise Exception(f'No Data Type: {data_type}')

    if algo == 'PCA':
        decomposition_algo = PCA(n_components=dim)
    elif algo == 'TruncatedSVD':
        decomposition_algo = TruncatedSVD(n_components=dim, algorithm='arpack')
    else:
        raise Exception(f'No Dimension Reduction Algorithm: {algo}')

    if eigen == 'Eigenvector':
        loadings = decomposition_algo.fit(data).components_.T
    elif eigen == 'Reduced Vector (Scaled data to [0, 1] based on the abs(loadings); Negative values are also meaningful)':
        loadings = decomposition_algo.fit_transform(data)
    else:
        raise Exception(f'Wrong')

    rotator = Rotator(method=rotation_method)
    loadings = rotator.fit_transform(loadings)

    if eigen == 'Reduced Vector (Scaled data to [0, 1] based on the abs(loadings); Negative values are also meaningful)':
        scaler = preprocessing.StandardScaler()
        loadings = scaler.fit_transform(loadings)
        scaler = preprocessing.MinMaxScaler()
        loadings = scaler.fit_transform(np.abs(loadings))
        # loadings = custom_scale(loadings)

    construct_names, construct_definition = [line[0] for line in data.index.tolist()], [line[1] for line in data.index.tolist()]
    return loadings, construct_names, construct_definition

# @st.cache_resource
def export_df(loadings, construct_names, construct_definition, cutoff):
    decimals = 4
    df = sort_rows_and_cutoff(loadings, construct_names, construct_definition, cutoff)
    df['ID'] = list([i+1 for i in range(len(df))])
    df = df.rename(columns={'index': 'Construct Name', 'Definition': 'Construct Definition'})
    df = df.set_index(['ID', 'Construct Name', 'Construct Definition'])
    for col in df.columns:
        df[col] = df[col].apply(lambda x: round_up(x, decimals) if isinstance(x, (int, float)) else x)

    return df

if __name__ == '__main__':
    model_option = st.selectbox(
        'Embedding LLM',
        ('WhereIsAI/UAE-Large-V1', 'avsolatorio/GIST-large-Embedding-v0', 'llmrails/ember-v1', 'Salesforce/SFR-Embedding-Mistral'))
    data_type = st.selectbox('Original Dataset', ('Correlation Matrix', 'Similarity Matrix'))
    dim_reduction_option = st.selectbox('Deimension Reduction Algorithm', ('PCA', 'TruncatedSVD'))
    eigen = st.selectbox('Eigenvector OR Reduced Vector', ('Eigenvector', 'Reduced Vector (Scaled data to [0, 1] based on the abs(loadings); Negative values are also meaningful)'))
    rotation_method = st.selectbox('Rotation Method', ('varimax', 'promax'))
    pca_dim_option = st.number_input('Dimension After Reduction', min_value=1, max_value=300, value=150, step=1)
    cutoff_option = st.number_input('Cut-Off Threshold', min_value=0., max_value=1., value=0.1)

    # prepare data
    option_prefix = {'WhereIsAI/UAE-Large-V1': 'uae-large-v1', 'avsolatorio/GIST-large-Embedding-v0': 'gist-large-0', 'llmrails/ember-v1': 'ember-v1', 'Salesforce/SFR-Embedding-Mistral': 'SFR-Embedding'}
    prefix = option_prefix[model_option]
    loadings, construct_names, construct_definition = dim_reduction_rotation(data_type, dim_reduction_option, rotation_method, prefix, pca_dim_option, eigen)
    df = export_df(loadings=loadings, construct_names=construct_names, construct_definition=construct_definition, cutoff=cutoff_option)

    # display the dataframe
    st.dataframe(df)

