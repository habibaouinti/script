import os
from io import StringIO
from scipy.io import mmread
import numpy as np


def convert_smtx_to_mtx(smtx_file, mtx_file=None):
    """
    Convert .smtx format to MatrixMarket .mtx format
    """
    if mtx_file is None:
        mtx_file = smtx_file.replace('.smtx', '.mtx')
    
    with open(smtx_file, 'r') as f_in, open(mtx_file, 'w') as f_out:
        # Read header
        header = f_in.readline().strip()
        nrows, ncols, nnz = map(int, header.split(','))
        
        # Write MatrixMarket header
        f_out.write('%%MatrixMarket matrix coordinate real general\n')
        f_out.write(f'{nrows} {ncols} {nnz}\n')
        
        # Process each row
        for row_idx in range(nrows):
            line = f_in.readline().strip()
            if not line:
                continue
                
            col_indices = list(map(int, line.split()))
            for col_idx in col_indices:
                # MatrixMarket uses 1-based indexing
                f_out.write(f'{row_idx + 1} {col_idx + 1} 1.0\n')
    
    # print(f'Converted {smtx_file} to {mtx_file}')
    # print(f'Matrix dimensions: {nrows} x {ncols}, Non-zeros: {nnz}')
    return mtx_file


def load_matrix(path_input_matrix="./", file_name=""):
    if ".smtx" in file_name:
        convert_smtx_to_mtx(os.path.join(path_input_matrix, file_name))
        file_name = file_name.replace(".smtx", ".mtx")
    if ".mtx" in file_name:
        matrix = load_matrice_mtx(path_input_matrix, file_name)
    elif ".npy" in file_name:
        matrix = load_matrice_npy(path_input_matrix, file_name)
    elif "pytorch" in file_name:
        ...
    else:
        raise Exception(f"Extension {file_name.split('.')[1]} is not supported.")
    return matrix

def load_matrice_mtx(path_matrix="./", name = "name.mtx"):
    name_path = os.path.join(path_matrix, name)

    with open(name_path) as f: text = f.read()
    m = mmread(StringIO(text))
    return m


def load_matrice_npy(path_matrix="./", name = "name.npy"):
    name_path = os.path.join(path_matrix, name)  
    m = np.load(name_path)
    return m


def get_size_mat(m):
    M,K = m.shape
    return M, K


def get_sparsity_ratio(m):
    return int((1-(m!=0).sum()/np.prod(m.shape)))*100