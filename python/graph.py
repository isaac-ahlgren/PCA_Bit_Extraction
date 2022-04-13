from bit_extract import tr_bit_extract
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from  multiprocessing import Process, Array, Manager
from multiprocessing.pool import Pool
import subprocess
import os
import time

import csv

MAX_PROCESSES = 7

def subprocesses_gen_shift_data(device_buffer, vector_length, bit_length, max_shift, filter_range, device,folder_name,repo_directory):

    if not os.path.exists(f"{repo_directory}/pickled_data/{folder_name}"):
        os.makedirs(f"{repo_directory}/pickled_data/{folder_name}")

    for shift in range(max_shift): 
        if not os.path.exists(f"{repo_directory}/pickled_data/{folder_name}/{device}_{shift}.npy"):
            device_samples = device_buffer[shift:(shift + vector_length*bit_length)]
            np.save(f"{repo_directory}/pickled_data/{folder_name}/{device}_{shift}", device_samples, allow_pickle=True)
    
    running_processes = 0
    processes = []
    index = -1
    shift = 0
    while shift < max_shift:
        if index == -1 and running_processes < MAX_PROCESSES:
            command = [f"python3 {repo_directory}/python/bit_extract.py {repo_directory} {repo_directory}/pickled_data/{folder_name}/{device}_{shift}.npy {bit_length} {filter_range} {device}_{shift} {folder_name}"]
            processes.append(subprocess.Popen(command, shell=True))
            running_processes+=1
            shift+=1
        elif index >= 0 and index < MAX_PROCESSES and running_processes < MAX_PROCESSES: 
            command = [f"python3 {repo_directory}/python/bit_extract.py {repo_directory} {repo_directory}/pickled_data/{folder_name}/{device}_{shift}.npy {bit_length} {filter_range} {device}_{shift} {folder_name}"]
            processes[index] = subprocess.Popen(command, shell=True)
            running_processes+=1
            shift+=1
            index = -1
        elif running_processes >= MAX_PROCESSES:
            for i in range(len(processes)):
                if processes[i].poll() != None:
                    index = i
                    running_processes -= 1
                    print(f"PID {processes[i].pid} finished.")
                    break

def gen_shift_data(host_buffer, device_buffer, vector_length, bit_length, max_shift, filter_range):
    stats = np.zeros(max_shift)
    host_samples = host_buffer[0:(vector_length*bit_length)]
    
    for shift in range(max_shift):
        device_samples = device_buffer[shift:(shift + vector_length*bit_length)]
        host_bits = tr_bit_extract(host_samples, bit_length, filter_range)
        device_bits = tr_bit_extract(device_samples, bit_length, filter_range)
        agreement_rate = compare_bits(host_bits, device_bits, bit_length)
        stats[shift] = agreement_rate

def pickle_it(filename, array):
    if not os.path.exists(filename + "_pickled"):
        np.save(filename + "_pickled.npy", array)

def graph(data, x_label, y_label, title_name, pdf_name, label_names):
    x = range(len(data[0,:]))
    data_plots = len(data[:,0])
    colors = ['#659DF6','#FF9300','#B4B4B4',"#666666"]
    for i in range(data_plots):
        plt.plot(x, data[i], color=colors[i], linewidth=0.7, label=label_names[i],)
    plt.rcParams["font.family"] = "serif"
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title_name)
    plt.rcParams["figure.figsize"] = (20,10)
    plt.rcParams.update({'font.size': 16})
    plt.grid(color='gainsboro', linestyle='--',linewidth=0.5,visible=True,which='minor',axis="y")
    plt.grid(color='gainsboro', linestyle='--',linewidth=0.5,visible=True,which='major',axis="y")
    plt.minorticks_on()
    plt.savefig(pdf_name + '.pdf') 
    plt.show()

def get_audio(directory, name):
    sr, data = wavfile.read(directory + "/" + name)
    normal_data = data / 32767
    return np.array(normal_data, dtype=np.float32)

def get_comparison_stats(bit_host_base, host_bit_directory, bit_other_base, other_bit_directory, bit_len, shift_len):
    stats = np.zeros(shift_len)    

    host_file = host_bit_directory + "/" + bit_host_base + "_0.csv"
    host_bits = np.zeros(bit_len)
    with open(host_file, newline='') as host:
        reader = csv.reader(host, delimiter=',', quotechar='|')
        buf = next(reader)
        for i in range(len(buf)):
            host_bits[i] = float(buf[i])

    for i in range(shift_len):
        dev_file = other_bit_directory + "/" + bit_other_base + "_" + str(i) + ".csv"
        with open(dev_file, newline='') as dev:
            bits = np.zeros(bit_len)
            reader = csv.reader(dev, delimiter=',', quotechar='|')
            buf = next(reader)
            print(buf)
            for j in range(len(buf)):
                bits[j] = float(buf[j])
            stats[i] = compare_bits(host_bits, bits, bit_len)

    return stats

def compare_bits(bits1, bits2, bit_length):
    agreed = 0
    for i in range(bit_length):
        if bits1[i] == bits2[i]:
            agreed += 1
    return (agreed/bit_length)*100

if __name__ == "__main__":
    # Parameters
    repo_directory = "/home/ikey/repos/PCA_Bit_Extraction"
    channels = 4
    obs_vector_length = 2000
    bit_key_length = 64
    max_shift = 5000
    filter_range = 5

    base_names = ["near_music", "medium_music", "far_music"]

    #file = repo_directory + "/data/electromagnetic_data/em_experiment.csv"
    #sample_len = 150000
    #buffers = np.zeros((channels,sample_len))
    #with open(file, newline='') as csvfile:
    #    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #    for j in range(sample_len):
    #        row = next(reader)
    #        for i in range(1,channels+1):
    #            buffers[i-1,j] = float(row[i])

    #device_names = ["55_cm", "50_cm", "35_cm", "25_cm"]
    #for i in range(channels):
    #    subprocesses_gen_shift_data(buffers[i,:], obs_vector_length, bit_key_length, max_shift, filter_range, device_names[i], device_names[i], repo_directory)

    data_directory = repo_directory + "/data/audio/wav/"
    for i in range(len(base_names)):
        track1_name = base_names[i] + "_track1.wav"
        track2_name = base_names[i] + "_track2.wav"
        track1 = get_audio(data_directory, track1_name)
        track2 = get_audio(data_directory, track2_name)
        subprocesses_gen_shift_data(track1, obs_vector_length, bit_key_length, max_shift, filter_range, base_names[i] + "_track1", base_names[i], repo_directory)
        subprocesses_gen_shift_data(track2, obs_vector_length, bit_key_length, max_shift, filter_range, base_names[i] + "_track2", base_names[i], repo_directory)

    stat_names = ["near_music", "medium_music", "far_music"] 
    comp_stats = np.zeros((len(stat_names),max_shift))
    for i in range(len(stat_names)):
        host_bit_directory = repo_directory + "/bit_results/" + stat_names[i]
        bit_host_base = stat_names[i] + "_track2"
        bit_other_base = stat_names[i] + "_track1"
        comp_stats[i,:] = get_comparison_stats(bit_host_base, host_bit_directory, bit_other_base, host_bit_directory, bit_key_length, max_shift)

    #label_names = ["55cm host and 50cm device", "55cm host and 35cm device", "35cm host and 25cm device"]
    #comp_stats = np.zeros((len(label_names),max_shift))

    #comp_stats[0,:] = get_comparison_stats("55_cm", repo_directory + "/bit_results/55_cm", "50_cm", repo_directory + "/bit_results/50_cm", bit_key_length, max_shift)
    #comp_stats[1,:] = get_comparison_stats("55_cm", repo_directory + "/bit_results/55_cm", "35_cm", repo_directory + "/bit_results/35_cm", bit_key_length, max_shift)
    #comp_stats[2,:] = get_comparison_stats("35_cm", repo_directory + "/bit_results/35_cm", "25_cm", repo_directory + "/bit_results/25_cm", bit_key_length, max_shift)

    graph(comp_stats, "Time Sample Shift", "Bit Agreement(%)", stat_names)
    
    
