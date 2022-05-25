import numpy as np
from scipy.spatial import distance
from graph import get_audio
from graph import graph
from graph import pickle_it
from graph import compare_bits
import ctypes
import matplotlib.pyplot as plt

def gen_pca_samples(x, vec_len, vec_num, eig_vecs, max_shift, pickle_name, directory):
    lib = ctypes.cdll.LoadLibrary("./distance_calc.so")   

    gen_pca_samples_c = lib.pca_shifted_calcs
    gen_pca_samples_c.restype = None
    gen_pca_samples_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]

    pca_samples = np.zeros(vec_num*eig_vecs*max_shift, dtype=np.float32)

    eig_vectors = np.zeros(int(vec_len/2 + 1)*eig_vecs*max_shift, dtype=np.float32)

    convergence = np.zeros(eig_vecs*max_shift, dtype=np.int32)

    gen_pca_samples_c(x, vec_len, vec_num, eig_vecs, max_shift, pca_samples, eig_vectors, convergence)

    split_pca_samples = np.array(np.split(pca_samples, max_shift))
    pca_samples = list()
    for i in range(max_shift):
        pca_samples.append(np.array(np.split(split_pca_samples[i], eig_vecs)))

    pickle_it(directory + pickle_name + "_pca_samples", np.array(pca_samples))

    split_eig_vectors = np.array(np.split(eig_vectors, max_shift))

    eig_vectors = list()
    for i in range(max_shift):
        eig_vectors.append(np.array(np.split(split_eig_vectors[i], eig_vecs)))

    pickle_it(directory + pickle_name + "_eigs", np.array(eig_vectors))

    split_convergence = np.array(np.split(convergence, eig_vecs))
    
    pickle_it(directory + pickle_name + "_conv", split_convergence)

def gen_rmse_fft_n_time(buf1, buf2, vec_len, shift_len):
    ret_fft = np.zeros(shift_len)
    ret_time = np.zeros(shift_len)

    host = buf1[0:vec_len]
    for i in range(shift_len):
        device = buf2[i: i + vec_len]
        ret_time[i] = np.sqrt(np.square(np.subtract(host,device)).mean())
        ret_fft[i] = np.sqrt(np.square(np.subtract(np.abs(np.fft.fft(host)),np.abs(np.fft.fft(device)))).mean())
    return ret_fft, ret_time

def gen_euclid_fft_n_time(buf1, buf2, vec_len, shift_len):
    ret_fft = np.zeros(shift_len)
    ret_time = np.zeros(shift_len)

    host = buf1[0:vec_len]
    for i in range(shift_len):
        device = buf2[i: i + vec_len]
        ret_time[i] = np.sqrt(np.square(np.subtract(host,device)).sum())
        ret_fft[i] = np.sqrt(np.square(np.subtract(np.abs(np.fft.fft(host)),np.abs(np.fft.fft(device)))).sum())
    return ret_fft, ret_time

def gen_cos_fft_n_time(buf1, buf2, vec_len, shift_len):
    ret_fft = np.zeros(shift_len)
    ret_time = np.zeros(shift_len)

    host = buf1[0:vec_len]
    for i in range(shift_len):
        device = buf2[i: i + vec_len]
        ret_time[i] = distance.cosine(host, device)
        ret_fft[i] = distance.cosine(np.abs(np.fft.fft(host)), np.abs(np.fft.fft(device)))
    return ret_fft, ret_time
 
if __name__ == "__main__":
   repo_directory = "/home/ikey/repos/PCA_Bit_Extraction"
   obs_vector_length = 2048
   vec_num = 64
   max_shift = 5000
   data_directory = repo_directory + "/data/audio"

   graph_directory = "./graphs/"

   directory = graph_directory + "pickled/"
   
   types = ["conversation", "cooking_audio", "music"]
   max_shift = 5000
   beg_pow2_len = 9
   end_pow2_len = 13
   beg_pow2_num = 5
   end_pow2_num = 9
   beg_pow2_eigs = 4
   end_pow2_eigs = 5
   beg_pow2_ds = 1
   end_pow2_ds = 4
   len_iterations = end_pow2_len - beg_pow2_len
   num_iterations = end_pow2_num - beg_pow2_num
   eig_iterations = end_pow2_eigs - beg_pow2_eigs
   ds_iterations = end_pow2_ds - beg_pow2_ds 
 
   for i in range(len(types)):
       vec_len = np.power(2, beg_pow2_len)
       for j in range(len_iterations):
           vec_num = np.power(2, beg_pow2_num)
           for k in range(num_iterations):
               eig_num = np.power(2, beg_pow2_eigs)
               for l in range(eig_iterations):
                   ds_num = np.power(2, beg_pow2_ds)
                   for m in range(ds_iterations):
                       for h in range(1,5):
                           track_name = "secured_" + types[i] + "_48khz_track" + str(h) + "_ds" + str(ds_num) +  ".wav"
                           track = get_audio(data_directory + "/secured/" + types[i], track_name)
                           name = types[i] + "_veclen" + str(vec_len) + "_vecnum" + str(vec_num) + "_eignum" + str(eig_num) + "_ds" + str(ds_num)
                           print(name)
                           gen_pca_samples(track, vec_len, vec_num, eig_num, max_shift, name, directory)
                   ds_num *= 2
               eig_num *= 2
           vec_num *= 2
       vec_len *= 2
   
   '''
   # Calc time and freq domain distances
   for i in range(len(types)):
       vec_len = np.power(2, beg_pow2)
       for j in range(iterations):
           res_fft_cos_dist = np.zeros((3, 5000))
           res_time_cos_dist = np.zeros((3, 5000))
           res_fft_euclid_dist = np.zeros((3, 5000))
           res_time_euclid_dist = np.zeros((3, 5000))
           track1_name = "secured_" + types[i] + "_48khz_track1.wav"
           track1 = get_audio(data_directory + "/" + types[i], track1_name)
           for k in range(2,5):
               track_name = "secured_" + types[i] + "_48khz_track" + str(k) + ".wav"
               track = get_audio(data_directory + "/" + types[i], track_name) 
               ret_fft_cos, ret_time_cos = gen_cos_fft_n_time(track1, track, vec_len, 5000)
               ret_fft_euclid, ret_time_euclid = gen_euclid_fft_n_time(track1, track, vec_len, 5000)
               res_fft_cos_dist[k-2,:] = ret_fft_cos
               res_time_cos_dist[k-2,:] = ret_time_cos
               res_fft_euclid_dist[k-2,:] = ret_fft_euclid
               res_time_euclid_dist[k-2,:] = ret_time_euclid
           graph(res_fft_cos_dist, "Time Sample Shifts", "Cosine Distance", titles[i] + " Frequency Domain Vector Length " + str(vec_len), graph_directory +  types[i] + "_cos_freq_"  + str(vec_len), labels)
           graph(res_time_cos_dist, "Time Sample Shifts", "Cosine Distance", titles[i] + " Time Domain Vector Length " + str(vec_len), graph_directory +  types[i] + "_cos_time_"  + str(vec_len), labels)
           graph(res_fft_euclid_dist, "Time Sample Shifts", "Euclidean Distance", titles[i] + " Frequency Domain Vector Length " + str(vec_len), graph_directory +  types[i] + "_euclid_freq_"  + str(vec_len), labels)
           graph(res_time_euclid_dist, "Time Sample Shifts", "Euclidean Distance", titles[i] + " Time Domain Vector Length " + str(vec_len), graph_directory +  types[i] + "_euclid_time_"  + str(vec_len), labels)
           vec_len = 2*vec_len

   data_directory = "/home/ikey/repos/PCA_Bit_Extraction/python/graphs/pickled/"

    # Calc PCA distances
   for i in range(len(types)):
       vec_len = np.power(2, beg_pow2)
       for j in range(iterations):
           res_cos = np.zeros((3, 5000))
           res_euclid = np.zeros((3, 5000))
           track1 = np.load(data_directory + "secured_" + types[i] + "_track1_veclen" + str(vec_len) + "_pickled.npy")
           for k in range(2,5):
               track = np.load(data_directory + "secured_" + types[i] + "_track1_veclen" + str(vec_len) + "_pickled.npy")
               res_cos[k-2,:] = gen_cos(track1, track)
               res_euclid[k-2,:] = gen_euclid(track1, track)
           graph(res_cos, "Time Sample Shifts", "Cosine Distance", titles[i] + " PCA Vector Length " + str(vec_len), graph_directory + types[i] + "_cos_pca_"  + str(vec_len), labels)
           graph(res_euclid, "Time Sample Shifts", "Euclidean Distance", titles[i] + " PCA Vector Length " + str(vec_len), graph_directory +  types[i] + "_euclid_pca_"  + str(vec_len), labels)
           vec_len = 2*vec_len

    # Calc keeping location same, vary vector sizes
   for i in range(len(types)):
       vec_len = np.power(2, beg_pow2)

       res_cos_12 = np.zeros((4, 5000))
       res_euclid_12 = np.zeros((4, 5000))
       res_cos_14 = np.zeros((4, 5000))
       res_euclid_14 = np.zeros((4, 5000))
       for j in range(iterations):
           track1 = np.load(data_directory + "secured_" + types[i] + "_track1_veclen" + str(vec_len) + "_pickled.npy")
           track2 = np.load(data_directory + "secured_" + types[i] + "_track2_veclen" + str(vec_len) + "_pickled.npy")
           track4 = np.load(data_directory + "secured_" + types[i] + "_track4_veclen" + str(vec_len) + "_pickled.npy")
           res_cos_12[j,:] = gen_cos(track1, track2)
           res_euclid_12[j,:] = gen_euclid(track1, track2)
           res_cos_14[j,:] = gen_cos(track1, track4)
           res_euclid_14[j,:] = gen_euclid(track1, track4)
           vec_len = 2*vec_len
       graph(res_cos_12, "Time Sample Shifts", "Cosine Distance", titles[i] + " PCA Near", graph_directory + types[i] + "_cos_near_pca_varvec", labels)
       graph(res_cos_14, "Time Sample Shifts", "Cosine Distance", titles[i] + " PCA Far" , graph_directory + types[i] + "_cos_far_pca_varvec", labels)
       graph(res_euclid_12, "Time Sample Shifts", "Euclidean Distance", titles[i] + " PCA Near", graph_directory + types[i] + "_euclid_near_pca_varvec", labels)
       graph(res_euclid_14, "Time Sample Shifts", "Euclidean Distance", titles[i] + " PCA Far" , graph_directory + types[i] + "_euclid_far_pca_varvec", labels)
       
   '''
