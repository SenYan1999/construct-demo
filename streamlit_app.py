import math
import pandas as pd
import numpy as np
import streamlit as st

from numpy.linalg import svd
from numpy import eye, asarray, dot, sum, diag
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from factor_analyzer.rotator import Rotator

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
    sorted_index = np.lexsort([-df_loadings[col] for col in reversed(df_loadings.columns)])
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
def dim_reduction_rotation(data_type, algo, rotation_method, prefix, dim):
    # load the data
    if data_type == 'Correlation Matrix':
        data = pd.read_csv(f'data/{prefix}_correlation_with_name.csv', index_col=[0,1])
    elif data_type == 'Similarity Matrix':
        data = pd.read_csv(f'data/{prefix}_similarity_with_name.csv', index_col=[0,1])
    else:
        raise Exception(f'No Data Type: {data_type}')

    if algo == 'PCA':
        pca = PCA(n_components=dim)
        # loadings = pca.fit_transform(data)
        loadings = pca.fit(data).components_.T
    elif algo == 'TruncatedSVD':
        svd = TruncatedSVD(n_components=dim, algorithm='arpack')
        # loadings = svd.fit_transform(data)
        loadings = svd.fit(data).components_.T
    else:
        raise Exception(f'No Dimension Reduction Algorithm: {algo}')

    if rotation_method == 'varimax':
        loadings = varimax(loadings)
    elif rotation_method == 'promax':
        rotator = Rotator(method='promax')
        loadings = rotator.fit_transform(loadings)
    else:
        raise Exception(f'No Rotation Method: {rotation_method}')

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
    rotation_method = st.selectbox('Rotation Method', ('varimax', 'promax'))
    pca_dim_option = st.number_input('Dimension After Reduction', min_value=1, max_value=300, value=150, step=1)
    cutoff_option = st.number_input('Cut-Off Threshold', min_value=0., max_value=1., value=0.1)

    # prepare data
    option_prefix = {'WhereIsAI/UAE-Large-V1': 'uae-large-v1', 'avsolatorio/GIST-large-Embedding-v0': 'gist-large-0', 'llmrails/ember-v1': 'ember-v1', 'Salesforce/SFR-Embedding-Mistral': 'SFR-Embedding'}
    prefix = option_prefix[model_option]
    loadings, construct_names, construct_definition = dim_reduction_rotation(data_type, dim_reduction_option, rotation_method, prefix, pca_dim_option)
    df = export_df(loadings=loadings, construct_names=construct_names, construct_definition=construct_definition, cutoff=cutoff_option)

    # display the dataframe
    st.dataframe(df)

