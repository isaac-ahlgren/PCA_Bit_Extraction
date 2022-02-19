import numpy as np
from scipy import linalg
import scipy.linalg.lapack as la
import scipy.fftpack

f1 = "/home/jweezy/Drive2/Drive2/Code/UC-Code/PCA_Bit_Extraction/pickled_data/doyle_500khz_2ndfloor_ds20/2ndfloor_0_98.npy"
f2 = "/home/jweezy/Drive2/Drive2/Code/UC-Code/PCA_Bit_Extraction/pickled_data/doyle_500khz_2ndfloor_ds20/2ndfloor_1_98.npy"
f3 = "/home/jweezy/Drive2/Drive2/Code/UC-Code/PCA_Bit_Extraction/pickled_data/doyle_500khz_2ndfloor_ds20/2ndfloor_2_98.npy"

S1 = np.load(f1)
S2 = np.load(f2)
S3 = np.load(f3)

vector_num = 64



def tr_bit_extract(data, vector_num):
    data_matrix = np.array(np.split(data, vector_num))
    
    vlen = len(data) // vector_num

    fft_data_matrix = np.abs(np.fft.fft(data_matrix))

    cov_matrix = np.cov(fft_data_matrix.T)

    return list(fft_data_matrix),list(cov_matrix)


fft1,cov1 = tr_bit_extract(S1, vector_num)

fft2,cov2 = tr_bit_extract(S2, vector_num)

fft3,cov3 = tr_bit_extract(S3, vector_num)


fft1 = matrix(SR, 64, 2000, fft1) 
fft2 = matrix(SR, 64, 2000, fft2)  
fft3 = matrix(SR, 64, 2000, fft3) 

print("Symbolic Matrix contstruction complete")

# for i in range(len(fft1.rows())):
#     for j in range(len(fft1.columns())):
#         fft1[i,j] = fft1[i,j]*var(f"x_{i}_{j}")
#         fft2[i,j] = fft2[i,j]*var(f"x_{i}_{j}")
#         fft3[i,j] = fft3[i,j]*var(f"x_{i}_{j}")

# print("Variables added to matrix entries") 

# symbolic_covariance_1 = fft1 * fft1.transpose()
# symbolic_covariance_2 = fft2 * fft2.transpose()
# symbolic_covariance_3 = fft3 * fft3.transpose()

# print("Symbolic covariance contstruction complete")

# res1 = symbolic_covariance_1-symbolic_covariance_2
# res2 = symbolic_covariance_1-symbolic_covariance_3

# print("Finished")

