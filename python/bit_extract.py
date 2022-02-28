from scipy import linalg
import scipy.linalg.lapack as la
import scipy.fftpack
import scipy
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
    data_matrix = np.array(np.split(data, vector_num))

    vlen = len(data) // vector_num

    fft_data_matrix = np.abs(np.fft.fft(data_matrix))

    #filter(fft_data_matrix, filter_range, vlen)

    cov_matrix = np.cov(fft_data_matrix.T)

    output = la.ssyevx(cov_matrix, range='I', il=vlen, iu=vlen)

    eig_vec = np.array(output[1])

    #eig_vec = fix_direction(eig_vec)

    proj_data = (eig_vec.T).dot(data_matrix.T)

    proj_data = proj_data[0]

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

    vector_num = 64

    filter_range = 0

    shift = args.shift

    folder_name = args.folder_name

  #  data = np.load("/home/ikey/repos/PCA_Bit_Extraction/pickled_data/near_room_ambient/near_room_ambient_track1_0.npy")
  #  data = data.astype(float) / 32767

  #  vector_num = 64

  #  filter_range = 0

  #  tr_bit_extract(data, vector_num, filter_range)
    tr_bit_extract_subprocess(data, vector_num, filter_range, shift, folder_name, repo_directory)
