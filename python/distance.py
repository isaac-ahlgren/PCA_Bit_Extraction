import numpy as np
from graph import get_audio
from graph import graph
from graph import pickle_it
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
        print("Calculating Vector Length " + str(vec_len))
        res_ep = np.zeros(max_shift, dtype=np.float32)
        res_cp = np.zeros(max_shift, dtype=np.float32)
        euclid_dist_pca_c(x, y, vec_len, vec_num, max_shift, res_ep)
        cosine_dist_pca_c(x, y, vec_len, vec_num, max_shift, res_cp)
        results_ep[i,:] = res_ep
        results_cp[i,:] = res_ep
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

def gen_hamming_dist(base_names, vec_len, vec_num, max_shift):
    so_file = "./distance_calc.so"
    lib = ctypes.cdll.LoadLibrary("./distance_calc.so")   

    gen_pca_samples_c = lib.gen_pca_samples
    gen_pca_samples_c.restype = None
    gen_pca_samples_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int]
      
    results = np.zeros((len(base_names), max_shift))
    pca_samples_t1 = np.zeros(vec_num)
    pca_samples_t2 = np.zeros(vec_num)

    for i in range(len(base_names)):
        track1_name = base_names[i] + "_track1.wav"
        track2_name = base_names[i] + "_track2.wav"
        track1 = get_audio(data_directory, track1_name)
        track2 = get_audio(data_directory, track2_name)
        for j in range(max_shift):
            gen_pca_samples_c(track1, pca_samples_t1, vec_len, vec_num, 0)
            gen_pca_samples_c(track2, pca_samples_t2, vec_len, vec_num, j)
            bits1 = gen_bits(pca_samples_t1)
            bits2 = gen_bits(pca_samples_t2)
            results[i,j] = compare_bits(bits1, bits2, vec_num)
    return results

def gen_hamming_dist_multiple_lens(base_names, label_names, pdf_base_name, vec_num, beg_pow2, end_pow2, max_shift):
    iterations = end_pow2 - beg_pow2
    for i in range(iterations):
        vec_len = np.power(2, beg_pow2)
        results = gen_hamming_dist(base_names, vec_len, vec_num, max_shift)
        graph(results, "Time Sample Shift", "Bit Agreement(%)", "", "./graphs/" + pdf_base_name + "_" + str(vec_len), label_names)

if __name__ == "__main__":
   '''
   so_file = "./distance_calc.so"
   lib = ctypes.cdll.LoadLibrary("./distance_calc.so")

   # Distance functions for regular
   euclid_dist_c = lib.euclid_dist_shift
   euclid_dist_c.restype = None
   euclid_dist_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                             np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                             ctypes.c_int,
                             ctypes.c_int,
                             np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

   cosine_dist_c = lib.cosine_dist_shift
   cosine_dist_c.restype = None
   cosine_dist_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                             np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                             ctypes.c_int,
                             ctypes.c_int,
                             np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

   levenshtein_dist_c = lib.levenshtein_dist_shift
   levenshtein_dist_c.restype = None
   levenshtein_dist_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                  ctypes.c_int,
                                  ctypes.c_int,
                                  np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

   # Distance functions for fft
   euclid_dist_fft_c = lib.euclid_dist_shift_fft
   euclid_dist_fft_c.restype = None
   euclid_dist_fft_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

   cosine_dist_fft_c = lib.cosine_dist_shift_fft
   cosine_dist_fft_c.restype = None
   cosine_dist_fft_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                 np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                 ctypes.c_int,
                                 ctypes.c_int,
                                 np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

   levenshtein_dist_fft_c = lib.levenshtein_dist_shift_fft
   levenshtein_dist_fft_c.restype = None
   levenshtein_dist_fft_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                      np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

   # Distance functions for pca
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

   levenshtein_dist_pca_c = lib.levenshtein_dist_shift_pca
   levenshtein_dist_pca_c.restype = None
   levenshtein_dist_pca_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                      np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
   '''

   repo_directory = "/home/ikey/repos/PCA_Bit_Extraction"
   obs_vector_length = 2048
   vec_num = 64
   max_shift = 5000
   data_directory = repo_directory + "/data/audio/wav"

   graph_directory = "./graphs/"

   # Vary Vector Sizes
   base_names = ["near_music", "medium_music", "far_music", "near_fire_ambient", "medium_fire_ambient", "far_fire_ambient", "near_room_ambient", "medium_room_ambient", "far_room_ambient"]
   labels = ["512 Length Vector", "1024 Length Vector", "2048 Length Vector", "4096 Length Vector"]
   
   results_ep = np.zeros((3, max_shift))
   results_cp = np.zeros((3, max_shift))
   for i in range(len(base_names)):
       track1_name = base_names[i] + "_track1.wav"
       track2_name = base_names[i] + "_track2.wav"
       track1 = get_audio(data_directory, track1_name)
       track2 = get_audio(data_directory, track2_name)

       results_ep, results_cp = multiple_windows_pca(track1, track2, vec_num, 9, 13, max_shift)

       pickle_it(graph_directory + "pickled/euclidian_" + base_names[i]  + "_pca", results_ep)
       pickle_it(graph_directory + "pickled/cosine_" + base_names[i]  + "_pca", results_cp)

       graph(results_ep, "Time Sample Shifts", "Euclidean Distance", "Euclidian Distance Over Multiple Vector Sizes", graph_directory + "euclidian_" + base_names[i]  + "_pca", labels)
       graph(results_cp, "Time Sample Shifts", "Cosine Distance", "Cosine Distance Over Multiple Vector Sizes", graph_directory + "cosine_" + base_names[i] + "_pca", labels)

   # Gen hamming distances for multiple vector sizes
   base_names = [["near_music", "medium_music", "far_music"], ["near_fire_ambient", "medium_fire_ambient", "far_fire_ambient"], ["near_room_ambient", "medium_room_ambient", "far_room_ambient"]]
   labels = [["Near Music", "Medium Music", "Far Music"], ["Near Fire Ambient", "Medium Fire Ambient", "Far Fire Ambient"], ["Near Room Ambient", "Medium Room Ambient", "Far Room Ambient"]]
   pdf_names = ["music_hamming_dist", "fire_hamming_dist", "room_hamming_dist"]
   for i in range(len(base_names)):
       gen_hamming_dist_multiple_lens(base_names[i], label_names[i], pdf_base_name[i], vec_num, 9, 13, max_shift)

   '''
    # Generate PCA data for music
   base_names = ["near_music", "medium_music", "far_music"]
   labels = ["Near Music", "Medium Music", "Far Music"]
   results_ep, results_cp = gen_pca_data_audio(base_names, obs_vector_length, vec_num, max_shift)
   graph(results_ep, "Time Sample Shifts", "Euclidean Distance", "", graph_directory + "euclidian_dist_music_pca", labels)
   graph(results_cp, "Time Sample Shifts", "Cosine Distance", "", graph_directory + "cosine_dist_music_pca", labels)

   # Generate PCA data for fire ambience
   base_names = ["near_fire_ambient", "medium_fire_ambient", "far_fire_ambient"]
   labels = ["Near Fire Ambience", "Medium Fire Ambience", "Far Fire Ambience"]
   results_ep, results_cp = gen_pca_data_audio(base_names, obs_vector_length, vec_num, max_shift)
   graph(results_ep, "Time Sample Shifts", "Euclidean Distance", "", graph_directory + "euclidian_dist_fire_ambient_pca", labels)
   graph(results_cp, "Time Sample Shifts", "Cosine Distance", "", graph_directory + "cosine_dist_fire_ambient_pca", labels)

   # Generate PCA data for room ambience
   base_names = ["near_room_ambient", "medium_room_ambient", "far_room_ambient"]
   labels = ["Near Room Ambience", "Medium Room Ambience", "Far Room Ambience"]
   results_ep, results_cp = gen_pca_data_audio(base_names, obs_vector_length, vec_num, max_shift)
   graph(results_ep, "Time Sample Shifts", "Euclidean Distance", "", graph_directory + "euclidian_dist_room_ambient_pca", labels)
   graph(results_cp, "Time Sample Shifts", "Cosine Distance", "", graph_directory + "cosine_dist_room_ambient_pca", labels)
   '''

   '''
   # Distances for room ambience
   base_names = ["near_room_ambient", "medium_room_ambient", "far_room_ambient"]
   labels = ["Near Room Ambience", "Medium Room Ambience", "Far Room Ambience"]
   #results_et = np.zeros((3, max_shift))
   #results_ef = np.zeros((3, max_shift))
   results_ep = np.zeros((3, max_shift))
   #results_ct = np.zeros((3, max_shift))
   #results_cf = np.zeros((3, max_shift))
   results_cp = np.zeros((3, max_shift))
   #results_lt = np.zeros((3, max_shift))
   #results_lf = np.zeros((3, max_shift))
   #results_lp = np.zeros((3, max_shift))
   for i in range(len(base_names)):
       track1_name = base_names[i] + "_track1.wav"
       track2_name = base_names[i] + "_track2.wav"
       track1 = get_audio(data_directory, track1_name)
       track2 = get_audio(data_directory, track2_name)

       #res_et = np.zeros(max_shift, dtype=np.float32)
       #res_ef = np.zeros(max_shift, dtype=np.float32)
       res_ep = np.zeros(max_shift, dtype=np.float32)
       #res_ct = np.zeros(max_shift, dtype=np.float32)
       #res_cf = np.zeros(max_shift, dtype=np.float32)
       res_cp = np.zeros(max_shift, dtype=np.float32)
       #res_lt = np.zeros(max_shift, dtype=np.float32)
       #res_lf = np.zeros(max_shift, dtype=np.float32)
       #res_lp = np.zeros(max_shift, dtype=np.float32)

       #euclid_dist_c(track1, track2, obs_vector_length, max_shift, res_et)
       #cosine_dist_c(track1, track2, obs_vector_length, max_shift, res_ct)
       #levenshtein_dist_c(track1, track2, obs_vector_length, max_shift, res_lt)
       #euclid_dist_fft_c(track1, track2, obs_vector_length, max_shift, res_ef)
       #cosine_dist_fft_c(track1, track2, obs_vector_length, max_shift, res_cf)
       #levenshtein_dist_fft_c(track1, track2, obs_vector_length, max_shift, res_lf)
       euclid_dist_pca_c(track1, track2, obs_vector_length, vec_num, max_shift, res_ep)
       cosine_dist_pca_c(track1, track2, obs_vector_length, vec_num, max_shift, res_cp)
       #levenshtein_dist_pca_c(track1, track2, obs_vector_length, vec_num, max_shift, res_lp)

       #results_et[i,:] = res_et
       #results_ef[i,:] = res_ef
       results_ep[i,:] = res_ep
       #results_ct[i,:] = res_ct
       #results_cf[i,:] = res_cf
       results_cp[i,:] = res_cp
       #results_lt[i,:] = res_lt
       #results_lf[i,:] = res_lf
       #results_lp[i,:] = res_lp

   #graph(results_et, "Time Sample Shifts", "Euclidian Distance", "euclidian_dist_room_ambient_time", labels)
   #graph(results_ef, "Time Sample Shifts", "Euclidian Distance", "euclidian_dist_room_ambient_fft", labels)
   graph(results_ep, "Time Sample Shifts", "Euclidian Distance", "euclidian_dist_room_ambient_pca", labels)
   #graph(results_ct, "Time Sample Shifts", "Cosine Distance", "cosine_dist_room_ambient_time", labels)
   #graph(results_cf, "Time Sample Shifts", "Cosine Distance", "cosine_dist_room_ambient_fft", labels)
   graph(results_cp, "Time Sample Shifts", "Cosine Distance", "cosine_dist_room_ambient_pca", labels)
   #graph(results_lt, "Time Sample Shifts", "Levenshtein Distance", "levenshtein_dist_room_ambient_time", labels)
   #graph(results_lf, "Time Sample Shifts", "Levenshtein Distance", "levenshtein_dist_room_ambient_fft", labels)
   #graph(results_lp, "Time Sample Shifts", "Levenshtein Distance", "levenshtein_dist_room_ambient_pca", labels)
   '''
