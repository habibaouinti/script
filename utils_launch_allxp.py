import random
import numpy as np
import struct;
from utils_gen_bitmal import load_matrice_mtx, load_matrice_npy, get_size_mat, get_sparsity_ratio
import os


random.seed(a=0, version=2)


def generate_random_B(N, K, rand_smallest=1, rand_largest=10, sparsity_ratio_b = 50, data_width = 4, dense_matrixB_file = "matrixB_dense.npy"):
    matrixB = []
    for _ in range(N):
        for _ in range(K):
            sparse_prob = random.randint(0, 100)
            if sparse_prob > sparsity_ratio_b:
                value = float(random.randint(rand_smallest, rand_largest))
                matrixB.append(value)
            else:
                matrixB.append(0.0)

    matrixB_np = np.array(matrixB, dtype=np.float32).reshape(N, K) 
    return  matrixB_np
    
    
# NAMES DNNs
# VGG16 3:1152,128, 4: 1152,256 and 9: 4608,512
# VDSR 9: 486,54 and 17, 14
# ResNet20:  14:576,64, 15: 516,64, 16: 576,64
# ResNet50:  2: 576,64, 11:256,64, 13:, 22:1152,128 
def get_b_dim_from_a(name_a, shape_a):
    N,K = 0, shape_a[1]
    # Read from the name the size of the activation input
    #  For VDSR we take a input of size HxW=800x800
    if "vdsr" in name_a.lower():
        # N,K = 0, 54*9
        N,K = 64516, 486
        
    #  For ResNet50 we take a input of ImageNet 1K so 300x400 ? 
    elif "resnet50" in name_a.lower():
        #  TODO vérifier les numéros
        layer_number = int(name_a.lower().split("_")[0])
        if layer_number == 2:
            N,K = 2916, 576
        elif layer_number == 11:
            N,K = 3136, 256
        elif layer_number == 22:
            N,K = 128, 1152
            
    #  For ResNet20 we take a input of CIFAR10 so 32x32 ? 
    elif "resnet20" in name_a.lower():
        #  TODO vérifier les numéros
        layer_number = int(name_a.lower().split("_")[0])
        if layer_number == 14:
            N,K = 36, 576
        elif layer_number == 15:
            N,K = 36, 576
        elif layer_number == 16:
            N,K = 36, 576
            
    #  For VGG16 we take a input of of ImageNet 1K so 300x400 ?   
    elif "vgg16" in name_a.lower():
        #  TODO vérifier les numéros
        layer_number = int(name_a.lower().split("_")[0])
        if layer_number == 3:
            # N,K = 576, 49284
            # N,K = 576, 12100
            # N,K = 1152, 2916
            N,K = 12100, 1152
        elif layer_number == 4:
            N,K = 2916, 1152
        elif layer_number == 9:
            N,K = 676, 4608
    else:
        raise Exception("No corresponding DNN. Check the names of the file")
    return N,K


def remove_ext(name=""):
    return name.replace(".npy","").replace(".mtx","")



def generate_meta_data(matrAx=None, file_nameA="", matrBx=None, file_nameB="", generate_result=1, test_output_file="result_test.out", data_width=4, path_file_meta_data=""):
    
    meta_data_txt = os.path.join(path_file_meta_data, "offset.txt")
    # read file A
    # if this is a mtx (suite sparse)
    # if ".mtx" in file_name:
    #     matrAx = load_matrice_mtx(path_input_matrix, file_name)
    # elif ".npy" in file_name:
    #     matrAx = load_matrice_npy(path_input_matrix, file_name)
    # else:
    #     raise Exception(f"Extension {file_name.split('.')[1]} is not supported.")

    # read file B
    # read if it's not empty
    generate_B = False
    # if len(file_nameB):
    #     generate_B = False
    #     # if this is a mtx (suite sparse)
    #     if ".mtx" in file_nameB:
    #         matrBx = load_matrice_mtx(path_input_matrixB, file_nameB)
    #     elif ".npy" in file_nameB:
    #         matrBx = load_matrice_npy(path_input_matrixB, file_nameB)
    #     elif "pytorch" in file_nameB:
    #         ...
    #     else:
    #         raise Exception(f"Extension {file_nameB.split('.')[1]} is not supported.")
    
    M,K = get_size_mat(matrAx)
    if generate_B:
        N = N;
    else:
        N,_ = get_size_mat(matrBx)

    sparsity_ratio_a=get_sparsity_ratio(matrAx)
    if generate_B:
        sparsity_ratio_b=sparsity_ratio_b
    else:
        sparsity_ratio_b=get_sparsity_ratio(matrBx)

    file_name=os.path.join(path_file_meta_data,f"bitmapSpMSpM_gemm_mem_{remove_ext(file_nameA)}_{remove_ext(file_nameB)}.ini")

    in_file_bitmap_a=os.path.join(path_file_meta_data,f"bitmapSpMSpM_file_bitmapA_{remove_ext(file_nameA)}_"+str(M)+"_"+str(N)+"_"+str(K)+".in")
    in_file_bitmap_b=os.path.join(path_file_meta_data,f"bitmapSpMSpM_file_bitmapB_{remove_ext(file_nameB)}_"+str(M)+"_"+str(N)+"_"+str(K)+".in")

    address_matrix_a=0;
    address_matrix_b=0; # to be updated in the code
    address_matrix_c=0; # to be updated in the code

    rand_smallest=1;
    rand_largest=10;

    if(generate_result):
        matrixA_size=int(M*K);
        matrixB_size=int(N*K);
        matrixC_size=int(M*N);

        matrixA=[]
        matrixB=[]
        matrixC=list(range(0,matrixC_size));
    random.seed(a=0, version=2)


    # Generating matrix A
    with open(file_name, "w") as fd, open(in_file_bitmap_a, "w") as fbA, open(in_file_bitmap_b, "w") as fbB:
        #generating matrixA
        n_nonzeros=0
        for m in range(M):  # Row major
            for k in range(K):
                # sparse_prob=random.randint(0,100);
                value=matrAx[m,k];
                if value:
                    if((m==(M-1)) and (k==(K-1))):
                        fbA.write(str(1))
                    else:
                        fbA.write(str(1)+","); #writing a 1 in bitmap
                    value = float(value)
                    ba = bytearray(struct.pack(">f", value))  # generating list of bytes
                    my_int = int.from_bytes(ba, "big")
                    fd.write(str(my_int))
                    fd.write(",")
                    n_nonzeros+=1;
                    if(generate_result):
                        matrixA.append(value);
                else:
                    if((m==(M-1)) and (k==(K-1))): # this is to insert a comma
                        fbA.write(str(0));
                        # note no data element is inserted in this case
                    else:
                        # note no data element is inserted in this case
                        fbA.write(str(0)+",");
                    if(generate_result):
                        matrixA.append(float(0.0));
                        
        address_matrix_b=n_nonzeros*data_width;
        #Generating matrix B
        n_nonzeros=0;
        bitmapB=list(range(0,matrixB_size));
        for n in range(0,N):  # Row major
            for k in range(0,K):
                if generate_B:
                    sparse_prob=random.randint(0,100);
                    if(sparse_prob > sparsity_ratio_b):  # value is generated
                        bitmapB[k*N+n]=1
                        value = float(random.randint(rand_smallest, rand_largest));
                        ba = bytearray(struct.pack(">f", value))  # generating list of bytes
                        my_int = int.from_bytes(ba, "big")
                        fd.write(str(my_int))
                        fd.write(",")
                        n_nonzeros+=1;
                        if(generate_result):
                            matrixB.append(value);
                    else:
                        # no data element is inserted in this case
                        bitmapB[k*N+n]=0; #writing a 0
                        if(generate_result):
                            matrixB.append(float(0.0));
                else:
                    value=matrBx[n,k];
                    if value:
                        bitmapB[k*N+n]=1
                        ba = bytearray(struct.pack(">f", value))  # generating list of bytes
                        my_int = int.from_bytes(ba, "big")
                        fd.write(str(my_int))
                        fd.write(",")
                        n_nonzeros+=1;
                        if(generate_result):
                            matrixB.append(value);
                    else:
                        # no data element is inserted in this case
                        bitmapB[k*N+n]=0; #writing a 0
                        if(generate_result):
                            matrixB.append(float(0.0));
                    
        # writing the bitmapB in the appropiate order
        for i in range(0, matrixB_size):
            fbB.write(str(bitmapB[i]));
            if(i < (matrixB_size-1)):
                fbB.write(",")
        
        fd.write(str(0)) # Adding a final 0 to the memory which will never be used. This is just to avoid having a last comma.
        address_matrix_c=address_matrix_b+(n_nonzeros*data_width);


    with open(meta_data_txt, "w") as meta_file:
        print("Offset matrix A: "+str(address_matrix_a), file=meta_file);
        print("Offset matrix B: "+str(address_matrix_b), file=meta_file);
        print("Offset matrix C: "+str(address_matrix_c), file=meta_file);

        # print("File "+file_name+" generated correctly", file=meta_file);
        # print("File "+in_file_bitmap_a+" generated correctly", file=meta_file);
        # print("File "+in_file_bitmap_b+" generated correctly", file=meta_file);
    return file_name, in_file_bitmap_a, in_file_bitmap_b, address_matrix_a, address_matrix_b, address_matrix_c
    # if(generate_result):
    #     for i in range(0, M ):
    #         for j in range(0, N):
    #             matrixC[i*N+j]=float(0.0)
    #             for k in range(0,K):
    #                 matrixC[i*N+j]+= matrixA[i*K+k]*matrixB[j*K+k] # row-major order in both matrices. (i.e., KN matrix is transposed)
    #     with open(os.path.join(path_file_meta_data,test_output_file), "w") as f:
    #         for i in range(0,matrixC_size):
    #             value = float(matrixC[i])
    #             f.write(str(value))
    #             f.write(",")


def generate_stonne_inner_product(path, accel_name, replace_name,K,N,M, file_name, in_file_bitmap_a, in_file_bitmap_b, address_matrix_a, address_matrix_b, address_matrix_c):
    
    a = f'# Define the simulation components\n'
    a += f'comp_stonne = sst.Component("stonne1", "sstStonne.MAERI")\n'
    a += 'comp_stonne.addParams({\n'
    a += f'    "hardware_configuration" : "sigma_128mses_128_bw.cfg",\n'
    a += f'    "kernelOperation" : "bitmapSpMSpM",\n'
    a += f'    "GEMM_K" : {K},\n'
    a += f'    "GEMM_N" : {N},\n'
    a += f'    "GEMM_M" : {M},\n'
    a += f'    "bitmap_matrix_a_init" : "{in_file_bitmap_a}",\n'
    a += f'    "bitmap_matrix_b_init" : "{in_file_bitmap_b}",\n'
    a += f'    "mem_init" : "{file_name}",\n'
    a += f'    "matrix_a_dram_address" : {address_matrix_a},\n'
    a += f'    "matrix_b_dram_address" : {address_matrix_b},\n'
    a += f'    "matrix_c_dram_address" : {address_matrix_c},\n'
    a += f'    "mem_matrix_c_file_name" : "result.out"\n'
    a += f'\n'
    a += '})\n'
    
    with open("sst_stonne_inner_product_m_gen") as fr, open(os.path.join(path,accel_name,replace_name), "w") as fw:
        text = fr.read()
        fw.write(text.replace('%replace%', a))
    
    
def generate_sh_test(path, accel_name):
    sh_name = f"launch_{accel_name}.sh"
    
    path_exe = os.path.join(path,"name_of_that_file.py")
    with open(sh_name, "w") as f:
        print(f"sst {path_exe}", file=f)
