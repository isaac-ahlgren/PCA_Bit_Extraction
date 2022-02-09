from bit_extract import tr_bit_extract
import numpy as np
import matplotlib.pyplot as plt
from  multiprocessing import Process, Array
import csv

def compare_bits(bits1, bits2, bit_length):
    agreed = 0
    for i in range(bit_length):
        if bits1[i] == bits2[i]:
            agreed += 1
    return (agreed/bit_length)*100

def gen_shift_data(host_buffer, device_buffer, vector_length, bit_length, max_shift, filter_range):
    stats = np.zeros(max_shift)
    for shift in range(max_shift):
        host_samples = host_buffer[0:(vector_length*bit_length)]
        device_samples = device_buffer[shift:(shift + vector_length*bit_length)]
        host_bits = tr_bit_extract(host_samples, bit_length, filter_range)
        device_bits = tr_bit_extract(device_samples, bit_length, filter_range)
        agreement_rate = compare_bits(host_bits, device_bits, bit_length)
        stats[shift] = agreement_rate

def thread_gen_shift_data(host_buffer, device_buffer, device_num,  vector_length, bit_length, max_shift, filter_range, stats):
    
    for shift in range(max_shift):
        host_samples = host_buffer[0:(vector_length*bit_length)]
        device_samples = device_buffer[shift:(shift + vector_length*bit_length)]
        print("device " + str(device_num) + ": " + str(shift))
        host_bits = tr_bit_extract(host_samples, bit_length, filter_range)
        device_bits = tr_bit_extract(device_samples, bit_length, filter_range)
        agreement_rate = compare_bits(host_bits, device_bits, bit_length)
        stats[device_num*max_shift + shift] = agreement_rate

def threaded_gen_shift_data(threads, host_buffer, device_buffers, vector_length, bit_length, max_shift, filter_range):
    shared_mem = Array("d", len(device_buffers)*max_shift)
    thread_list = list()
    for i in range(threads):
        thread_list.append(Process(target=thread_gen_shift_data, args=(host_buffer,device_buffers[i],i,vector_length,bit_length,max_shift,filter_range, shared_mem,)))

    for j in range(threads):
        thread_list[j].start()

    for k in range(threads):
        thread_list[k].join()

    stats = np.zeros((len(device_buffers), max_shift))

    for i in range(len(device_buffers)):
        stats[i,:] = shared_mem[i*max_shift:(i+1)*max_shift]

    return stats

def graph(data, x_label, y_label, label_names):
    x = range(len(data[0,:]))
    data_plots = len(data[:,0])
    for i in range(data_plots):
        plt.plot(x, data[i], label=label_names[i])
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
        

if __name__ == "__main__":
    # Parameters
    file = "./"
    channels = 3
    obs_vector_length = 2000
    bit_key_length = 64
    max_shift = 5000
    filter_range = 100

    sample_len = 200000
    buffers = np.zeros((channels,sample_len))
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for j in range(sample_len):
            row = next(reader)
            for i in range(1,channels+1):
                buffers[i-1,j] = float(row[i])
  
    stats = gen_shift_data(buffers[0], buffers, obs_vector_length, bit_key_length, max_shift, filter_range)
    #stats = threaded_gen_shift_data(3, buffers[0], buffers, obs_vector_length, bit_key_length, max_shift, filter_range) 
    label_names = ["Third Floor 1", "Third Floor 2", "ML Hallway"]
    graph(stats, "Sample Shifts", "Bit Agreement",label_names)

