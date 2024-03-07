import math
import pandas as pd
import numpy as np
import streamlit as st

from numpy.linalg import svd
from numpy import eye, asarray, dot, sum, diag
from sklearn.decomposition import PCA

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
    rotated_loadings[rotated_loadings < threshold] = 0

    # Convert to DataFrame for easier manipulation, setting row names as the index
    df_loadings = pd.DataFrame(rotated_loadings, index=construct_names)

    # Sort rows in descending order based on loadings, prioritizing earlier columns
    sorted_index = np.lexsort([-df_loadings[col] for col in reversed(df_loadings.columns)])
    df_loadings.insert (0, "Definition", construct_definitions)
    df_sorted = df_loadings.iloc[sorted_index]

    # Reset index if you want the row names to become a regular column in the Excel file
    df_sorted.reset_index(inplace=True)
    df_sorted.replace(0, '', inplace=True)
    return df_sorted

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

@st.cache_resource
def pca(prefix, dim):
    data = pd.read_csv(f'data/{prefix}_similarity_with_name.csv', index_col=[0,1])
    fa = PCA(n_components=dim)
    fa.fit(data)
    loadings = fa.components_.T
    loadings = varimax(loadings)

    construct_names, construct_definition = [line[0] for line in data.index.tolist()], [line[1] for line in data.index.tolist()]
    return loadings, construct_names, construct_definition

@st.cache_resource
def export_df(loadings, construct_names, construct_definition, cutoff):
    decimals = 4
    df = sort_rows_and_cutoff(loadings, construct_names, construct_definition, cutoff)
    df = df.rename(columns={'index': 'Construct Name', 'Definition': 'Construct Definition'})
    df = df.set_index(['Construct Name', 'Construct Definition'])
    for col in df.columns:
        df[col] = df[col].apply(lambda x: round_up(x, decimals) if isinstance(x, (int, float)) else x)

    return df

if __name__ == '__main__':
    model_option = st.selectbox(
        'Embedding LLM',
        ('WhereIsAI/UAE-Large-V1', 'avsolatorio/GIST-large-Embedding-v0', 'llmrails/ember-v1', 'Salesforce/SFR-Embedding-Mistral'))
    pca_dim_option = st.number_input('Dimension After PCA', min_value=1, max_value=300, value=150, step=1)
    cutoff_option = st.number_input('Cut-Off Threshold', min_value=0., max_value=1., value=0.1)

    # prepare data
    option_prefix = {'WhereIsAI/UAE-Large-V1': 'uae-large', 'avsolatorio/GIST-large-Embedding-v0': 'gist-large', 'llmrails/ember-v1': 'emberv1', 'Salesforce/SFR-Embedding-Mistral': 'SFR'}
    prefix = option_prefix[model_option]
    loadings, construct_names, construct_definition = pca(prefix, pca_dim_option)
    df = export_df(loadings=loadings, construct_names=construct_names, construct_definition=construct_definition, cutoff=cutoff_option)

    # display the dataframe
    st.dataframe(df)

