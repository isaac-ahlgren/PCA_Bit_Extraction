from scipy import linalg
import scipy.linalg.lapack as la
import scipy.fftpack
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("repo_directory", help="file path for repo")
parser.add_argument("data", help="path to pickeled file")
parser.add_argument("vector_num", help="The size of the matrix")
parser.add_argument("filter_range", help="The range to be filtered.")
parser.add_argument("shift", help="This is not as imporant and will be used to give a file a name.")

# Data is a numpy array
def tr_bit_extract_subprocess(repo_directory, data, vector_num, filter_range,shift):
    
    bits = tr_bit_extract(data, vector_num, filter_range)

    with open(f"{repo_directory}/bit_results/{shift}.csv", "w") as f:
        for i in range(len(bits)):
            bit = bits[i]
            if i < len(bits)-1:
                f.write(f"{bit},")
            else:
                f.write(f"{bit}")

# Data is a numpy array
def tr_bit_extract(data, vector_num, filter_range):
    data_matrix = np.array(np.split(data, vector_num))
    
    vlen = len(data) // vector_num
    
    fft_data_matrix = np.abs(np.fft.fft(data_matrix))
        
    normalize(fft_data_matrix)

#    plot_fft(fft_data_matrix, fft_data_matrix.shape[0], 1/vlen)

    cov_matrix = np.cov(fft_data_matrix.T)

    output = la.ssyevx(cov_matrix, range='I', il=vlen, iu=vlen)

    eig_vec = np.array(output[1])

#    plot_1d_array(eig_vec.T, eig_vec.T.shape[1])

    eig_vec = fix_direction(eig_vec)

    proj_data = (eig_vec.T).dot(fft_data_matrix.T)

#    plot_1d_array(proj_data, proj_data[0].shape[0])

    proj_data = proj_data[0]

    return gen_bits(proj_data)

def filter(data, filter_range, vlen):
    for i in range(filter_range):
        data[:,i] = 0
        data[:,vlen-i-1] = 0

def normalize(data_matrix):
   m = np.mean(data_matrix,axis=1)

   for i in range(data_matrix.shape[0]):
       for j in range(data_matrix.shape[1]):
           data_matrix[j:i] -= m[i]

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
    return abs(eig_vec)

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

    tr_bit_extract_subprocess(repo_directory, data, vector_num, filter_range,shift)
    

