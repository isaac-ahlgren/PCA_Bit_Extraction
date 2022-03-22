import numpy as np
from graph import get_audio
from graph import graph
from bit_extract import tr_bit_extract
import ctypes

def euclid_dist(x,y):
    total = 0
    for i in range(len(x)):
        total += np.square(x[i] - y[i])

    return np.sqrt(total)

def cosine_dist(x,y):
    total = 0
    for i in range(len(x)):
       total += x[i]*y[i]
   
    return total

def levenshtein_dist(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]

def gen_euclid_dist_shift(x, y, sample_len, max_shift):
    res = np.zeros(max_shift)
    for shift in range(max_shift):
        buf1 = x[0:sample_len]
        buf2 = y[shift:(shift + sample_len)]
        res[shift] = euclid_dist(buf1, buf2)
    return res

def gen_euclid_dist_fft_shift(x, y, sample_len, max_shift):
    res = np.zeros(max_shift)
    for shift in range(max_shift):
        buf1 = np.abs(np.fft.fft(x[0:sample_len]))
        buf2 = np.abs(np.fft.fft(y[shift:(shift + sample_len)]))
        res[shift] = euclid_dist(buf1, buf2)
    return res

if __name__ == "__main__":
   so_file = "./distance_calc.so"
   lib = ctypes.cdll.LoadLibrary("./distance_calc.so")
   euclid_dist_c = lib.euclid_dist_shift
   euclid_dist_c.restype = None
   euclid_dist_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                             np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                             ctypes.c_int,
                             ctypes.c_int,
                             np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

   euclid_dist_fft_c = lib.euclid_dist_shift_fft
   euclid_dist_fft_c.restype = None
   euclid_dist_fft_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                ctypes.c_int,
                                ctypes.c_int,
                                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

   repo_directory = "/home/ikey/repos/PCA_Bit_Extraction"
   obs_vector_length = 2048
   max_shift = 5000
   data_directory = repo_directory + "/data/audio/wav"

   base_names = ["near_music", "medium_music", "far_music"]
   labels = ["Near Music", "Medium Music", "Far Music"]
   results = np.zeros((3, max_shift))
   results2 = np.zeros((3, max_shift))
   for i in range(len(base_names)):
       track1_name = base_names[i] + "_track1.wav"
       track2_name = base_names[i] + "_track2.wav"
       track1 = get_audio(data_directory, track1_name)
       track2 = get_audio(data_directory, track2_name)
       res = np.zeros(max_shift, dtype=np.float32)
       euclid_dist_c(track1, track2, obs_vector_length, max_shift, res)
       res2 = gen_euclid_dist_shift(track1, track2, obs_vector_length, max_shift)
       results[i,:] = res
       results2[i,:] = res2

   graph(results, "Time Sample Shifts", "Euclidian Distance", "euclidan_dist_music_c", labels)
   graph(results, "Time Sample Shifts", "Euclidian Distance", "euclidan_dist_music_python", labels)

   base_names = ["near_music", "medium_music", "far_music"]
   labels = ["Near Music", "Medium Music", "Far Music"]
   results = np.zeros((3, max_shift))
   results2 = np.zeros((3, max_shift))
   for i in range(len(base_names)):
       track1_name = base_names[i] + "_track1.wav"
       track2_name = base_names[i] + "_track2.wav"
       track1 = get_audio(data_directory, track1_name)
       track2 = get_audio(data_directory, track2_name)
       res = np.zeros(max_shift, dtype=np.float32)
       euclid_dist_fft_c(track1, track2, obs_vector_length, max_shift, res)
       res2 = gen_euclid_dist_fft_shift(track1, track2, obs_vector_length, max_shift)
       results[i,:] = res
       results2[i,:] = res2

   graph(results, "Time Sample Shifts", "Euclidian Distance", "euclidan_dist_fft_music_c", labels)
   graph(results, "Time Sample Shifts", "Euclidian Distance", "euclidan_dist_fft_music_python", labels)


#   base_names = ["near_fire_ambient", "medium_fire_ambient", "far_fire_ambient"]
#   labels = ["Near Fire Ambience", "Medium Fire Ambience", "Far Fire Ambience"]
#   results = np.zeros((3, max_shift))
#   for i in range(len(base_names)):
#       track1_name = base_names[i] + "_track1.wav"
#       track2_name = base_names[i] + "_track2.wav"
#       track1 = get_audio(data_directory, track1_name)
#       track2 = get_audio(data_directory, track2_name)
#       res = np.zeros(max_shift)
#       euclid_dist_c(track1, track2, obs_vector_length, max_shift, res)
#       results[i,:] = res

#   graph(results, "Time Sample Shifts", "Euclidian Distance", "euclidan_dist_fire_ambience", labels)

#   base_names = ["near_room_ambient", "medium_room_ambient", "far_room_ambient"]
#   labels = ["Near Room Ambience", "Medium Room Ambience", "Far Room Ambience"]
#   results = np.zeros((3, max_shift))
#   for i in range(len(base_names)):
#       track1_name = base_names[i] + "_track1.wav"
#       track2_name = base_names[i] + "_track2.wav"
#       track1 = get_audio(data_directory, track1_name)
#       track2 = get_audio(data_directory, track2_name)
#       res = np.zeros(max_shift)
#       euclid_dist_c(track1, track2, obs_vector_length, max_shift, res)
#       results[i,:] = res

#   graph(results, "Time Sample Shifts", "Euclidian Distance", "euclidan_dist_room_ambience", labels)
