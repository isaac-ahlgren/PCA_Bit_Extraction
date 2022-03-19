import numpy as np
from graph import get_audio
from graph import graph
from bit_extract import tr_bit_extract

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

def gen_dist(buf1, buf2, obs_vector_len, max_shift):
    dist_time = np.zeros((3,max_shift))
    dist_fft = np.zeros((3,max_shift))
    for shift in range(max_shift):
        dev_samples1 = buf1[0:obs_vector_len]
        dev_samples2 = buf2[shift:(shift + obs_vector_len)]
        dist_time[0,shift] = euclid_dist(dev_samples1,dev_samples2)
        dist_fft[0,shift] = euclid_dist(np.abs(np.fft.fft(dev_samples1)), np.abs(np.fft.fft(dev_samples2)))
        dist_time[1,shift] = cosine_dist(dev_samples1,dev_samples2)
        dist_fft[1,shift] = cosine_dist(np.abs(np.fft.fft(dev_samples1)), np.abs(np.fft.fft(dev_samples2)))
        dist_time[2,shift] = levenshtein_dist(dev_samples1,dev_samples2)
        dist_fft[2,shift] = levenshtein_dist(np.abs(np.fft.fft(dev_samples1)), np.abs(np.fft.fft(dev_samples2)))

    return dist_fft, dist_time

if __name__ == "__main__":
   repo_directory = ".."
   obs_vector_length = 2000
   max_shift = 5000
   data_directory = repo_directory + "/data/wav"
   base_names = ["near_music", "far_music"]
   distance_labels = ["Euclidian Distance", "Cosine Distance", "Levenshtein Distance"]
   for i in range(len(base_names)):
       track1_name = base_names[i] + "_track1.wav"
       track2_name = base_names[i] + "_track2.wav"
       track1 = get_audio(data_directory, track1_name)
       track2 = get_audio(data_directory, track2_name)
       dist_fft, dist_time = gen_dist(track1, track2, obs_vector_length, max_shift)
       graph(dist_time, "Time Sample Shifts", "Distance", base_names[i], distance_labels)
       graph(dist_fft, "Time Sample Shifts", "Distance", base_names[i], distance_labels)


