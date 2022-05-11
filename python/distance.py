import numpy as np
from scipy.spatial import distance
from graph import get_audio
from graph import graph
from graph import pickle_it
from graph import compare_bits
from bit_extract import tr_bit_extract
from bit_extract import gen_bits
import ctypes
import matplotlib.pyplot as plt

def multiple_windows_pca(x, y, vec_num, beg_pow2, end_pow2, max_shift):
    so_file = "./distance_calc.so"
    lib = ctypes.cdll.LoadLibrary("./distance_calc.so")

    euclid_dist_pca_c = lib.euclid_dist_shift_pca
    euclid_dist_pca_c.restype = None
    euclid_dist_pca_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

    cosine_dist_pca_c = lib.cosine_dist_shift_pca
    cosine_dist_pca_c.restype = None
    cosine_dist_pca_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

    iterations = end_pow2 - beg_pow2

    results_ep = np.zeros((iterations, max_shift))
    results_cp = np.zeros((iterations, max_shift))
   
    vec_len = np.power(2, beg_pow2)
    for i in range(iterations):
        print("Calculating the Vector Length " + str(vec_len))
        res_ep = np.zeros(max_shift, dtype=np.float32)
        res_cp = np.zeros(max_shift, dtype=np.float32)
        #euclid_dist_pca_c(x, y, vec_len, vec_num, max_shift, res_ep)
        cosine_dist_pca_c(x, y, vec_len, vec_num, max_shift, res_cp)
        results_ep[i,:] = res_ep
        results_cp[i,:] = res_cp
        vec_len = 2*vec_len

    return results_ep, results_cp

def gen_pca_data_audio(base_names, vec_len, vec_num, max_shift):
    so_file = "./distance_calc.so"
    lib = ctypes.cdll.LoadLibrary("./distance_calc.so")

    euclid_dist_pca_c = lib.euclid_dist_shift_pca
    euclid_dist_pca_c.restype = None
    euclid_dist_pca_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

    cosine_dist_pca_c = lib.cosine_dist_shift_pca
    cosine_dist_pca_c.restype = None
    cosine_dist_pca_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

    results_ep = np.zeros((len(base_names), max_shift))
    results_cp = np.zeros((len(base_names), max_shift))

    for i in range(len(base_names)):
        track1_name = base_names[i] + "_track1.wav"
        track2_name = base_names[i] + "_track2.wav"
        track1 = get_audio(data_directory, track1_name)
        track2 = get_audio(data_directory, track2_name)

        res_ep = np.zeros(max_shift, dtype=np.float32)
        res_cp = np.zeros(max_shift, dtype=np.float32)

        euclid_dist_pca_c(track1, track2, vec_len, vec_num, max_shift, res_ep)
        cosine_dist_pca_c(track1, track2, vec_len, vec_num, max_shift, res_cp)

        results_ep[i,:] = res_ep
        results_cp[i,:] = res_cp
       
    return results_ep, results_cp

def gen_pca_samples(x, vec_len, vec_num, max_shift, pickle_name):
    lib = ctypes.cdll.LoadLibrary("./distance_calc.so")   

    gen_pca_samples_c = lib.pca_shifted_calcs
    gen_pca_samples_c.restype = None
    gen_pca_samples_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

    results = np.zeros(vec_num*max_shift, dtype=np.float32)

    gen_pca_samples_c(x, vec_len, vec_num, max_shift, results)

    split_results = np.array(np.split(results, max_shift))

    pickle_it(pickle_name, split_results)

def gen_bit_extract_graphs(buf1, buf2):
    shift_len = len(buf1[:,0])
    bit_len = len(buf1[0,:])

    hamming_dist = np.zeros(shift_len)
    
    host_bits = gen_bits(buf1[0,:])

    for i in range(shift_len):
        device_bits = gen_bits(buf2[i,:])
        hamming_dist[i] = compare_bits(host_bits, device_bits, bit_len)
    return hamming_dist

def gen_rmse(buf1, buf2):
    shift_len = len(buf1[:,0])
    bit_len = len(buf1[0,:])

    ret = np.zeros(shift_len)

    host = buf1[0,:]

    for i in range(shift_len):
        device = buf2[i,:]
        ret[i] = np.sqrt(np.square(np.subtract(host,device)).mean())
    return ret

def gen_cos(buf1, buf2):
    shift_len = len(buf1[:,0])
    bit_len = len(buf1[0,:])

    ret = np.zeros(shift_len)

    host = buf1[0,:]

    for i in range(shift_len):
        device = buf2[i,:]
        ret[i] = distance.cosine(host, device)
    return ret

def gen_euclid(buf1, buf2):
    shift_len = len(buf1[:,0])
    bit_len = len(buf1[0,:])

    ret = np.zeros(shift_len)

    host = buf1[0,:]

    for i in range(shift_len):
        device = buf2[i,:]
        ret[i] = np.sqrt(np.square(np.subtract(host,device)).sum())
    return ret


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
   data_directory = "../audio/"

   graph_directory = "./graphs/"

   directory = graph_directory + "pickled/"
   
   vec_labels = ["512 Length Vector", "1024 Length Vector", "2048 Length Vector", "4096 Length Vector"]
   labels = ["Near", "Medium", "Far"]
   
   s_and_us = ["secured", "unsecured"]
   types = ["conversation", "cooking_audio", "music", "room_audio"]
   max_shift = 5000
   vec_num = 64
   beg_pow2 = 13
   end_pow2 = 15
   iterations = end_pow2 - beg_pow2

   for i in range(len(s_and_us)):
       for j in range(len(types)):
           for k in range(1,5):
               track_name = s_and_us[i] + "/" + types[j] + "/" + s_and_us[i] + "_" + types[j] + "_48khz_track" + str(k) + ".wav"
               track = get_audio(data_directory, track_name)
               
               vec_len = np.power(2, beg_pow2)
               for h in range(iterations):
                   print(s_and_us[i] + "_" + types[j] + "_track" + str(k) + "_veclen" + str(vec_len))
                   gen_pca_samples(track, vec_len, vec_num, max_shift, directory + s_and_us[i] + "_" + types[j] + "_track" + str(k) + "_veclen" + str(vec_len))
                   vec_len = 2*vec_len

           
   

   data_directory = data_directory + "/secured"
   labels = ["Near", "Medium", "Far"]
   titles = ["Conversation", "Cooking Ambient", "Music", "Room Ambient"]
   types = ["conversation", "cooking_audio", "music", "room_audio"]
   max_shift = 5000
   vec_num = 64
   beg_pow2 = 9
   end_pow2 = 13
   iterations = end_pow2 - beg_pow2
   vec_num = 64
    
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
