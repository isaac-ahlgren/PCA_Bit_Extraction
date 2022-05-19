from ctypes import *
from graph import get_audio
from graph import compare_bits
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import os

# Data is a numpy array
def tr_bit_extract(data, vector_len, vector_num, eigen_vectors):
    eig_vecs = np.zeros((eigen_vectors, int(vector_len/2) + 1))

    data_matrix = data[0:vector_len*vector_num]

    data_matrix = np.array(np.split(data_matrix, int(vector_num)))

    fft_data_matrix = np.abs(np.fft.rfft(data_matrix))

    cov_matrix = np.cov(fft_data_matrix.T)

    w, v = np.linalg.eigh(cov_matrix)

    for i in range(eigen_vectors):
        eig_vecs[i,:] = v[:, len(v) - 1 - i]

    fixed_eig_vecs = fix_direction(eig_vecs)

    proj_data = (fixed_eig_vecs).dot(fft_data_matrix.T)

    bits = gen_bits(proj_data)

    return bits, proj_data, fixed_eig_vecs, eig_vecs

def gen_bits(proj_data):
    proj_vectors = len(proj_data[:,0])
    vector_len = len(proj_data[0,:])
    bits = np.zeros(proj_vectors*vector_len)
    aves = np.zeros(proj_vectors)

    for i in range(proj_vectors):
        aves[i] = np.median(proj_data[i,:])

    for i in range(vector_len):
        for j in range(proj_vectors):
            if proj_data[j,i] > aves[j]:
               bits[i*proj_vectors + j] = 1;
            else:
                bits[i*proj_vectors + j] = 0;
    return bits

def fix_direction(eig_vecs):
    eig_vec_num = len(eig_vecs[:,0])
    vector_len = len(eig_vecs[0,:])
    print(eig_vec_num)
    print(vector_len)
    for j in range(eig_vec_num):
        total = 0
        for i in range(vector_len):
            total += eig_vecs[j,i]
    
        if total < 0:
            eig_vecs[j,:] *= -1
     
    return eig_vecs

def fix_direction_abs(eig_vec):
    return np.abs(eig_vec)

if __name__ == "__main__":
     vector_len = 4096
     vector_num = 256
     eig_vectors = 4
     data_directory = "/home/ikey/repos/PCA_Bit_Extraction/data/audio"
     data1 = get_audio(data_directory, "secured/cooking_audio/secured_cooking_audio_48khz_track1.wav")
     data2 = get_audio(data_directory, "secured/cooking_audio/secured_cooking_audio_48khz_track2.wav")
     data4 = get_audio(data_directory, "secured/cooking_audio/secured_cooking_audio_48khz_track4.wav")
     bits1, proj_data1, fixed_eig_vecs1, eig_vecs1 = tr_bit_extract(data1, vector_len, vector_num, eig_vectors)
     bits2, proj_data2, fixed_eig_vecs2, eig_vecs2 = tr_bit_extract(data2, vector_len, vector_num, eig_vectors)
     bits4, proj_data4, fixed_eig_vec4, eig_vecs4 = tr_bit_extract(data4, vector_len, vector_num, eig_vectors)

     print(eig_vecs1)
     print(eig_vecs2)
     print("Tracks 1 and 2")
     print("Bit Agreement: " + str(compare_bits(bits1, bits2, eig_vectors*vector_num)))
     print(bits1)
     print(bits2)
     print("PCA Samples Cosine Similarity: " + str(distance.cosine(proj_data1.flatten(), proj_data2.flatten())))
     print("PCA Samples Euclidean Distance: " + str(distance.euclidean(proj_data1.flatten(), proj_data2.flatten())))
     print()
     print("Tracks 1 and 4")
     print("Bit Agreement: " + str(compare_bits(bits1, bits4, eig_vectors*vector_num)))
     print(bits1)
     print(bits4)
     print("PCA Samples Cosine Similarity: " + str(distance.cosine(proj_data1.flatten(), proj_data4.flatten())))
     print("PCA Samples Euclidean Distance: " + str(distance.euclidean(proj_data1.flatten(), proj_data4.flatten())))

