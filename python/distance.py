import numpy as np
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
    so_file = "./distance_calc.so"
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

    split_results = np.array(np.split(results, vec_num))
    print(pickle_name)
    print(results)
    print(split_results)
    pickle_it(pickle_name, split_results)

def gen_bit_extract_graphs(buf1, buf2):
    shift_len = len(buf1[0,:])
    bit_len = len(buf1[:,0])
    print(shift_len)
    print(bit_len)
    hamming_dist = np.zeros(shift_len)
    
    host_bits = gen_bits(buf1[:,0])
    print(buf1)
    print(buf2)
    for i in range(shift_len):
        device_bits = gen_bits(buf2[:,i])
        hamming_dist[i] = compare_bits(host_bits, device_bits, bit_len)
    return hamming_dist
    
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
   max_shift = 2
   data_directory = repo_directory + "/data/audio/wav"

   graph_directory = "./graphs/"

   directory = graph_directory + "pickled/"
   
   
   vec_labels = ["512 Length Vector", "1024 Length Vector", "2048 Length Vector", "4096 Length Vector"]
   labels = ["Near", "Medium", "Far"]
   '''
   # Room Ambient
   #euclid_fra = np.load(directory + "euclidian_far_room_ambient_pca_pickled.npy")
   cosine_fra = np.load(directory + "cosine_far_room_ambient_pca_pickled.npy")
   #euclid_mra = np.load(directory + "euclidian_medium_room_ambient_pca_pickled.npy")
   cosine_mra = np.load(directory + "cosine_medium_room_ambient_pca_pickled.npy")
   #euclid_nra = np.load(directory + "euclidian_near_room_ambient_pca_pickled.npy")
   cosine_nra = np.load(directory + "cosine_near_room_ambient_pca_pickled.npy")

   #euclid_ra_512 = np.array([euclid_nra[0,:], euclid_mra[0,:], euclid_fra[0,:]])
   cosine_ra_512 = np.array([cosine_nra[0,:], cosine_mra[0,:], cosine_fra[0,:]])
   #euclid_ra_1024 = np.array([euclid_nra[1,:], euclid_mra[1,:], euclid_fra[1,:]])
   cosine_ra_1024 = np.array([cosine_nra[1,:], cosine_mra[1,:], cosine_fra[1,:]])
   #euclid_ra_2048 = np.array([euclid_nra[2,:], euclid_mra[2,:], euclid_fra[2,:]])
   cosine_ra_2048 = np.array([cosine_nra[2,:], cosine_mra[2,:], cosine_fra[2,:]])
   #euclid_ra_4096 = np.array([euclid_nra[3,:], euclid_mra[3,:], euclid_fra[3,:]])
   cosine_ra_4096 = np.array([cosine_nra[3,:], cosine_mra[3,:], cosine_fra[3,:]])

   #graph(euclid_fra, "Time Sample Shifts", "Euclidean Distance", "Far Room Ambience Over Different Vector Sizes", graph_directory + "euclidian_far_room_ambient_pca", vec_labels)
   graph(cosine_fra, "Time Sample Shifts", "Cosine Distance", "Far Room Ambience Over Different Vector Sizes", graph_directory + "cosine_far_room_ambient_pca", vec_labels)
   #graph(euclid_mra, "Time Sample Shifts", "Euclidean Distance", "Medium Room Ambience Over Different Vector Sizes", graph_directory + "euclidian_medium_room_ambient_pca", vec_labels)
   graph(cosine_mra, "Time Sample Shifts", "Cosine Distance", "Medium Room Ambience Over Different Vector Sizes", graph_directory + "cosine_medium_room_ambient_pca", vec_labels)
   #graph(euclid_nra, "Time Sample Shifts", "Euclidean Distance", "Near Room Ambience Over Different Vector Sizes", graph_directory + "euclidian_near_room_ambient_pca", vec_labels)
   graph(cosine_nra, "Time Sample Shifts", "Cosine Distance", "Near Room Ambience Over Different Vector Sizes", graph_directory + "cosine_near_room_ambient_pca", vec_labels)

   #graph(euclid_ra_512, "Time Sample Shifts", "Euclidean Distance", "Room Ambience (Vector Size 512)", graph_directory + "euclidian_room_ambient_pca_512", labels)
   graph(cosine_ra_512, "Time Sample Shifts", "Cosine Distance", "Room Ambience (Vector Size 512)", graph_directory + "cosine_far_room_ambient_pca_512", labels)
   #graph(euclid_ra_1024, "Time Sample Shifts", "Euclidean Distance", "Room Ambience (Vector Size 1024)", graph_directory + "euclidian_room_ambient_pca_1024", labels)
   graph(cosine_ra_1024, "Time Sample Shifts", "Cosine Distance", "Room Ambience (Vector Size 1024)", graph_directory + "cosine_medium_room_ambient_pca_1024", labels)
   #graph(euclid_ra_2048, "Time Sample Shifts", "Euclidean Distance", "Room Ambience (Vector Size 2048)", graph_directory + "euclidian_room_ambient_pca_2048", labels)
   graph(cosine_ra_2048, "Time Sample Shifts", "Cosine Distance", "Room Ambience (Vector Size 2048)", graph_directory + "cosine_near_room_ambient_pca_2048", labels)
   #graph(euclid_ra_4096, "Time Sample Shifts", "Euclidean Distance", "Room Ambience (Vector Size 4096)", graph_directory + "euclidian_room_ambient_pca_4096", labels)
   graph(cosine_ra_4096, "Time Sample Shifts", "Cosine Distance", "Room Ambience (Vector Size 4096)", graph_directory + "cosine_near_room_ambient_pca_4096", labels)

   # Music
   #euclid_fm = np.load(directory + "euclidian_far_music_pca_pickled.npy")
   cosine_fm = np.load(directory + "cosine_far_music_pca_pickled.npy")
   #euclid_mm = np.load(directory + "euclidian_medium_music_pca_pickled.npy")
   cosine_mm = np.load(directory + "cosine_medium_music_pca_pickled.npy")
   #euclid_nm = np.load(directory + "euclidian_near_music_pca_pickled.npy")
   cosine_nm = np.load(directory + "cosine_near_music_pca_pickled.npy")

   #euclid_m_512 = np.array([euclid_nm[0,:], euclid_mm[0,:], euclid_fm[0,:]])
   cosine_m_512 = np.array([cosine_nm[0,:], cosine_mm[0,:], cosine_fm[0,:]])
   #euclid_m_1024 = np.array([euclid_nm[1,:], euclid_mm[1,:], euclid_fm[1,:]])
   cosine_m_1024 = np.array([cosine_nm[1,:], cosine_mm[1,:], cosine_fm[1,:]])
   #euclid_m_2048 = np.array([euclid_nm[2,:], euclid_mm[2,:], euclid_fm[2,:]])
   cosine_m_2048 = np.array([cosine_nm[2,:], cosine_mm[2,:], cosine_fm[2,:]])
   #euclid_m_4096 = np.array([euclid_nm[3,:], euclid_mm[3,:], euclid_fm[3,:]])
   cosine_m_4096 = np.array([cosine_nm[3,:], cosine_mm[3,:], cosine_fm[3,:]])

   #graph(euclid_fm, "Time Sample Shifts", "Euclidean Distance", "Far Music Over Different Vector Sizes", graph_directory + "euclidian_far_music_pca", vec_labels)
   graph(cosine_fm, "Time Sample Shifts", "Cosine Distance", "Far Music Over Different Vector Sizes", graph_directory + "cosine_far_music_pca", vec_labels)
   #graph(euclid_mm, "Time Sample Shifts", "Euclidean Distance", "Medium Music Over Different Vector Sizes", graph_directory + "euclidian_medium_music_pca", vec_labels)
   graph(cosine_mm, "Time Sample Shifts", "Cosine Distance", "Medium Music Over Different Vector Sizes", graph_directory + "cosine_medium_music_pca", vec_labels)
   #graph(euclid_nm, "Time Sample Shifts", "Euclidean Distance", "Near Music Over Different Vector Sizes", graph_directory + "euclidian_near_music_pca", vec_labels)
   graph(cosine_nm, "Time Sample Shifts", "Cosine Distance", "Near Music Over Different Vector Sizes", graph_directory + "cosine_near_music_pca", vec_labels)

   #graph(euclid_m_512, "Time Sample Shifts", "Euclidean Distance", "Music (Vector Size 512)", graph_directory + "euclidian_music_pca_512", labels)
   graph(cosine_m_512, "Time Sample Shifts", "Cosine Distance", "Music (Vector Size 512)", graph_directory + "cosine_music_pca_512", labels)
   #graph(euclid_m_1024, "Time Sample Shifts", "Euclidean Distance", "Music (Vector Size 1024)", graph_directory + "euclidian_music_pca_1024", labels)
   graph(cosine_m_1024, "Time Sample Shifts", "Cosine Distance", "Music (Vector Size 1024)", graph_directory + "cosine_music_pca_1024", labels)
   #graph(euclid_m_2048, "Time Sample Shifts", "Euclidean Distance", "Music (Vector Size 2048)", graph_directory + "euclidian_music_pca_2048", labels)
   graph(cosine_m_2048, "Time Sample Shifts", "Cosine Distance", "Music (Vector Size 2048)", graph_directory + "cosine_music_pca_2048", labels)
   #graph(euclid_m_4096, "Time Sample Shifts", "Euclidean Distance", "Music (Vector Size 4096)", graph_directory + "euclidian_music_pca_4096", labels)
   graph(cosine_m_4096, "Time Sample Shifts", "Cosine Distance", "Music (Vector Size 4096)", graph_directory + "cosine_music_pca_4096", labels)

   #Fire Ambient
   #euclid_ffa = np.load(directory + "euclidian_far_fire_ambient_pca_pickled.npy")
   cosine_ffa = np.load(directory + "cosine_far_fire_ambient_pca_pickled.npy")
   #euclid_mfa = np.load(directory + "euclidian_medium_fire_ambient_pca_pickled.npy")
   cosine_mfa = np.load(directory + "cosine_medium_fire_ambient_pca_pickled.npy")
   #euclid_nfa = np.load(directory + "euclidian_near_fire_ambient_pca_pickled.npy")
   cosine_nfa = np.load(directory + "cosine_near_fire_ambient_pca_pickled.npy")

   #euclid_fa_512 = np.array([euclid_nfa[0,:], euclid_mfa[0,:], euclid_ffa[0,:]])
   cosine_fa_512 = np.array([cosine_nfa[0,:], cosine_mfa[0,:], cosine_ffa[0,:]])
   #euclid_fa_1024 = np.array([euclid_nfa[1,:], euclid_mfa[1,:], euclid_ffa[1,:]])
   cosine_fa_1024 = np.array([cosine_nfa[1,:], cosine_mfa[1,:], cosine_ffa[1,:]])
   #euclid_fa_2048 = np.array([euclid_nfa[2,:], euclid_mfa[2,:], euclid_ffa[2,:]])
   cosine_fa_2048 = np.array([cosine_nfa[2,:], cosine_mfa[2,:], cosine_ffa[2,:]])
   #euclid_fa_4096 = np.array([euclid_nfa[3,:], euclid_mfa[3,:], euclid_ffa[3,:]])
   cosine_fa_4096 = np.array([cosine_nfa[3,:], cosine_mfa[3,:], cosine_ffa[3,:]])

   #graph(euclid_ffa, "Time Sample Shifts", "Euclidean Distance", "Far Fire Ambience Over Different Vector Sizes", graph_directory + "euclidian_far_fire_ambient_pca", vec_labels)
   graph(cosine_ffa, "Time Sample Shifts", "Cosine Distance", "Far Fire Ambience Over Different Vector Sizes", graph_directory + "cosine_far_fire_ambient_pca", vec_labels)
   #graph(euclid_mfa, "Time Sample Shifts", "Euclidean Distance", "Medium Fire Ambience Over Different Vector Sizes", graph_directory + "euclidian_medium_fire_ambient_pca", vec_labels)
   graph(cosine_mfa, "Time Sample Shifts", "Cosine Distance", "Medium Fire Ambience Over Different Vector Sizes", graph_directory + "cosine_medium_fire_ambient_pca", vec_labels)
   #graph(euclid_nfa, "Time Sample Shifts", "Euclidean Distance", "Near Fire Ambience Over Different Vector Sizes", graph_directory + "euclidian_near_fire_ambient_pca", vec_labels)
   graph(cosine_nfa, "Time Sample Shifts", "Cosine Distance", "Near Fire Ambience Over Different Vector Sizes", graph_directory + "cosine_near_fire_ambient_pca", vec_labels)

   #graph(euclid_fa_512, "Time Sample Shifts", "Euclidean Distance", "Fire Ambience (Vector Size 512)", graph_directory + "euclidian_fire_ambient_pca_512", labels)
   graph(cosine_fa_512, "Time Sample Shifts", "Cosine Distance", "Fire Ambience (Vector Size 512)", graph_directory + "cosine_far_fire_ambient_pca_512", labels)
   #graph(euclid_fa_1024, "Time Sample Shifts", "Euclidean Distance", "Fire Ambience (Vector Size 1024)", graph_directory + "euclidian_fire_ambient_pca_1024", labels)
   graph(cosine_fa_1024, "Time Sample Shifts", "Cosine Distance", "Fire Ambience (Vector Size 1024)", graph_directory + "cosine_medium_fire_ambient_pca_1024", labels)
   #graph(euclid_fa_2048, "Time Sample Shifts", "Euclidean Distance", "Fire Ambience (Vector Size 2048)", graph_directory + "euclidian_fire_ambient_pca_2048", labels)
   graph(cosine_fa_2048, "Time Sample Shifts", "Cosine Distance", "Fire Ambience (Vector Size 2048)", graph_directory + "cosine_near_fire_ambient_pca_2048", labels)
   #graph(euclid_fa_4096, "Time Sample Shifts", "Euclidean Distance", "Fire Ambience (Vector Size 4096)", graph_directory + "euclidian_fire_ambient_pca_4096", labels)
   graph(cosine_fa_4096, "Time Sample Shifts", "Cosine Distance", "Fire Ambience (Vector Size 4096)", graph_directory + "cosine_near_fire_ambient_pca_4096", labels)
    
   '''
   '''
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

       #pickle_it(graph_directory + "pickled/euclidian_" + base_names[i]  + "_pca", results_ep)
       pickle_it(graph_directory + "pickled/cosine_" + base_names[i]  + "_pca", results_cp)

       #graph(results_ep, "Time Sample Shifts", "Euclidean Distance", "Euclidian Distance Over Multiple Vector Sizes", graph_directory + "euclidian_" + base_names[i]  + "_pca", labels)
       #graph(results_cp, "Time Sample Shifts", "Cosine Distance", "Cosine Distance Over Multiple Vector Sizes", graph_directory + "cosine_" + base_names[i] + "_pca", labels)
   '''
   
   base_names = ["near_music", "medium_music", "far_music", "near_fire_ambient", "medium_fire_ambient", "far_fire_ambient", "near_room_ambient", "medium_room_ambient", "far_room_ambient"]
   beg_pow2 = 9
   end_pow2 = 13
   iterations = end_pow2 - beg_pow2
   vec_num = 64

   for i in range(len(base_names)):
       track1_name = base_names[i] + "_track1.wav"
       track2_name = base_names[i] + "_track2.wav"
       track1 = get_audio(data_directory, track1_name)
       track2 = get_audio(data_directory, track2_name)

       vec_len = np.power(2, beg_pow2)
       for j in range(iterations):
           pickle_name1 = "./graphs/pickled/" + base_names[i] + "_track1_" + "veclen" + str(vec_len) + "_vecnum" + str(vec_num) +  "_maxshift" + str(max_shift)
           pickle_name2 = "./graphs/pickled/" + base_names[i] + "_track2_" + "veclen" + str(vec_len) + "_vecnum" + str(vec_num) +  "_maxshift" + str(max_shift)
           gen_pca_samples(track1, vec_len, vec_num, max_shift, pickle_name1)
           gen_pca_samples(track2, vec_len, vec_num, max_shift, pickle_name2)
           vec_len = 2*vec_len
   '''
   names = ["far_room_ambient_track1_veclen2048_vecnum64_maxshift5000_pickled.npy", "far_room_ambient_track2_veclen2048_vecnum64_maxshift5000_pickled.npy", "medium_room_ambient_track1_veclen2048_vecnum64_maxshift5000_pickled.npy", "medium_room_ambient_track2_veclen2048_vecnum64_maxshift5000_pickled.npy", "near_room_ambient_track1_veclen2048_vecnum64_maxshift5000_pickled.npy", "near_room_ambient_track2_veclen2048_vecnum64_maxshift5000_pickled.npy"]

   res = np.zeros((3, 5000))
   for i in range(3):
       buf1 = np.load(directory + names[2*i])
       buf2 = np.load(directory + names[2*i + 1])
       res[i,:] = gen_bit_extract_graphs(buf1, buf2)

   graph(res, "Time Sample Shifts", "Bit Agreement(%)", "Room Ambience Bit Agreement", graph_directory + "room_ambient_bit_agreement", labels)
   '''
