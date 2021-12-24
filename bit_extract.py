from scipy import linalg
import scipy.linalg.lapack as la
import numpy as np

# Data is a numpy array
def tr_bit_extract(data, vector_num, filter_range):
    data_matrix = np.array(np.split(data, vector_num))
    vlen = len(data) // vector_num

    fft_data_matrix = np.abs(np.fft.fft(data_matrix))

    filter(fft_data_matrix, filter_range, vlen)

    cov_matrix = np.cov(fft_data_matrix.T)

    output = la.ssyevx(cov_matrix, range='I', il=vlen, iu=vlen)

    eig_vec = np.array(output[1])

    proj_data = (eig_vec.T).dot(data_matrix.T)

    proj_data = proj_data[0]

    return gen_bits(proj_data)

def filter(data, filter_range, vlen):
    for i in range(filter_range):
        data[:,i] = 0
        data[:,vlen-i-1] = 0

def gen_bits(proj_data):
    med = np.median(proj_data)
    bits = np.zeros(len(proj_data))
    for i in range(len(proj_data)):
        if proj_data[i] > med:
            bits[i] = 1;
        else:
            bits[i] = 0;
    return bits
