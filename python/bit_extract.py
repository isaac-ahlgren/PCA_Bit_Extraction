from scipy import linalg
import scipy.linalg.lapack as la
import scipy.fftpack
import scipy
from ctypes import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("repo_directory", help="file path for repo")
parser.add_argument("data", help="path to pickeled file")
parser.add_argument("vector_num", help="The size of the matrix")
parser.add_argument("filter_range", help="The range to be filtered.")
parser.add_argument("shift", help="This is not as imporant and will be used to give a file a name.")
parser.add_argument("folder_name", help="This is not as imporant and will be used to give a file a name.")

# Data is a numpy array
def tr_bit_extract_subprocess(data, vector_num, filter_range, shift, folder_name, repo_directory):

    bits = tr_bit_extract(data, vector_num, filter_range)

    if not os.path.exists(f"{repo_directory}/bit_results/{folder_name}"):
        os.makedirs(f"{repo_directory}/bit_results/{folder_name}")

    with open(f"{repo_directory}/bit_results/{folder_name}/{shift}.csv", "w") as f:
        for i in range(len(bits)):
            bit = bits[i]
            if i < len(bits)-1:
                f.write(f"{bit},")
            else:
                f.write(f"{bit}")

# Data is a numpy array
def tr_bit_extract(data, vector_num, filter_range):

    data_matrix = np.array(np.split(data, int(vector_num)))

    vlen = len(data) // int(vector_num)

    fft_data_matrix = np.abs(np.fft.fft(data_matrix))

    #plot_fft(fft_data_matrix,fft_data_matrix.shape[0], 1/vlen)

    #filter(fft_data_matrix, filter_range, vlen)

    #plot_fft(fft_data_matrix,fft_data_matrix.shape[0], 1/vlen)

    cov_matrix = np.cov(fft_data_matrix.T)

    output = la.ssyevx(cov_matrix, range='I', il=vlen, iu=vlen)

    eig_vec = np.array(output[1])

    eig_vec = fix_direction_abs(eig_vec)

    proj_data = (eig_vec.T).dot(fft_data_matrix.T)[0]

    bits = gen_bits(proj_data)

    return bits

def filter(data, filter_range, vlen):
    for i in range(filter_range):
        data[:,i] = 0
        data[:,vlen-i-1] = 0

def normalize(data_matrix):
   m = np.mean(data_matrix,axis=1)

   for i in range(data_matrix.shape[0]):
       for j in range(data_matrix.shape[1]):
           data_matrix[j:i] -= m[i]

def mod_filter(fft_mat, mod):
    for i in range(len(fft_mat[:,0])):
        for j in range(len(fft_mat[0,:])):
           fft_mat[i,j] = c_float(c_uint(fft_mat[i,j]) % mod)

def gen_bits(proj_data):
    med = np.median(proj_data)
    bits = np.zeros(len(proj_data))
    for i in range(len(proj_data)):
        if proj_data[i] > med:
            bits[i] = 1;
        else:
            bits[i] = 0;
    return bits

def fix_direction(eig_vec):
    total = 0
    for i in range(len(eig_vec)):
        total += eig_vec[i]
    
    if total < 0:
        eig_vec *= -1
     
    return eig_vec

def fix_direction_abs(eig_vec):
    return np.abs(eig_vec)

# Plotting Functions for Debugging
def plot_fft(y,N,T):
    x = np.linspace(0.0, N*T, N)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.show()

def plot_1d_array(y,len):
    print(len)
    X = [i for i in range(2000)]

    plt.plot(X,y[0])
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()

    repo_directory = args.repo_directory

    data = args.data
    data = np.load(data)

    vector_num = int(args.vector_num)

    filter_range = int(args.filter_range)

    shift = args.shift

    folder_name = args.folder_name

 #   vector_num = 64

#   filter_range = 5

#    data1 = np.load("/home/ikey/repos/PCA_Bit_Extraction/pickled_data/55_cm/55_cm_0.npy")
#    data2 = np.load("/home/ikey/repos/PCA_Bit_Extraction/pickled_data/50_cm/50_cm_0.npy")
#    data3 = np.load("/home/ikey/repos/PCA_Bit_Extraction/pickled_data/35_cm/35_cm_0.npy")
#    data4 = np.load("/home/ikey/repos/PCA_Bit_Extraction/pickled_data/25_cm/25_cm_0.npy")

#    fft_data1 = np.abs(np.fft.fft(data1))
#    fft_data2 = np.abs(np.fft.fft(data2))

#    fft_data1 = np.array(np.split(fft_data1, int(vector_num)))
#    fft_data2 = np.array(np.split(fft_data2, int(vector_num)))

#    plt.plot(fft_data1)
#    plt.show()
#    plt.plot(fft_data2)
#    plt.show()
#    plt.plot(np.abs(np.fft.fft(data3)))
#    plt.show()
#    plt.plot(np.abs(np.fft.fft(data4)))
#    plt.show()

#    mod_filter(fft_data1, 100)
#    mod_filter(fft_data2, 100)

#    plt.plot(fft_data1)
#    plt.show()
#    plt.plot(fft_data2)
#    plt.show()

    #tr_bit_extract(data1, vector_num, filter_range)

    tr_bit_extract_subprocess(data, vector_num, filter_range, shift, folder_name, repo_directory)
