import os
from utils_gen_bitmal import load_matrix, get_size_mat, get_sparsity_ratio
from utils_launch_allxp import generate_random_B, get_b_dim_from_a, generate_meta_data, generate_stonne_inner_product, generate_sh_test
import numpy as np


accelerators = ["Flexigon_O","Flexigon_I", "Flexigon_G", "Sigma", "TPU", "MAERI", "SWAN"]
# Read accelerator
for accelerator in accelerators:
    ...
accel_name = "Flexigon_I"
path_dir_matrix = "./matrices/"
# Iterate over all the matrix in the directory
# Generate the flow for each matrix A and B
for nameA in os.listdir(path_dir_matrix):
    # Read the matrix A
    matrixA = load_matrix(path_dir_matrix, nameA)    
    M, K = matrixA.shape
    sparsity_a = get_sparsity_ratio(matrixA)
    # Take the dimension of B
    # If it's a DNN take the size of the activation tensor of N-1
    if ".npy" in nameA:
        N, K = get_b_dim_from_a(nameA, matrixA.shape)
    # If it's a sparse suite dataset take the same size as the matrix A
    elif not ".npy" in nameA:
        N, K = get_size_mat(matrixA)
    # Generate the matrix B
    # For two sparsity B 0 and 30
    for sparsity_b in [0,30]:
        dense_matrixB_file = f"matrixB_dense_{sparsity_b}.npy"
        data_width = 16
        rand_smallest = 1
        rand_largest = 10
        matrixB = generate_random_B(N, K, rand_smallest=rand_smallest, rand_largest=rand_largest, sparsity_ratio_b = sparsity_b, data_width=data_width, dense_matrixB_file=dense_matrixB_file)
        
        if ".npy" in nameA:
            name_a = nameA.replace(".npy","")
        else:
            name_a = nameA.replace(".mtx","")
        path = os.path.join("results", name_a, str(sparsity_a), f"MatrixB_{N}{K}",str(sparsity_b))
        if not os.path.isdir(path):
                os.makedirs(path)
        np.save(os.path.join(path,"MatrixA.npy"), matrixA)
        np.save(os.path.join(path,"MatrixB.npy"), matrixB)
        file_name, in_file_bitmap_a, in_file_bitmap_b, address_matrix_a, address_matrix_b, address_matrix_c = generate_meta_data(matrAx=matrixA, file_nameA="MatrixA.npy", \
                           matrBx=matrixB, file_nameB="MatrixB.npy", generate_result=1, \
                           test_output_file="result_test.out", data_width=data_width, path_file_meta_data=path)
        # Generating the parameter file
        if not os.path.isdir(os.path.join(path,accel_name)):
                os.makedirs(os.path.join(path,accel_name))
        generate_stonne_inner_product(path, accel_name, "sst_stonne_inner_product_m.py",K,N,M, file_name, in_file_bitmap_a, in_file_bitmap_b, address_matrix_a, address_matrix_b, address_matrix_c)
        
        # For stonne generate the metadata for A and B
        # Generate the path
        generate_sh_test(path, accel_name)





