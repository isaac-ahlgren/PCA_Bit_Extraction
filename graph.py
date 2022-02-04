from bit_extract import tr_bit_extract
import numpy as np
import matplotlib.pyplot as plt
from  multiprocessing import Process
import csv

def compare_bits(bits1, bits2, bit_length):
    agreed = 0
    for i in range(bit_length):
        if bits1[i] == bits2[i]:
            agreed += 1
    return (agreed/bit_length)*100

def gen_shift_data(host_buffer, device_buffer, device_num,  vector_length, bit_length, max_shift, filter_range, stats):
    
    for shift in range(max_shift):
        host_samples = host_buffer[0:(vector_length*bit_length)]
        device_samples = device_buffer[shift:(shift + vector_length*bit_length)]
        print("device " + str(device_num) + ": " + str(shift))
        host_bits = tr_bit_extract(host_samples, bit_length, filter_range)
        device_bits = tr_bit_extract(device_samples, bit_length, filter_range)
        agreement_rate = compare_bits(host_bits, device_bits, bit_length)
        stats[device_num,shift] = agreement_rate
    return stats

def threaded_gen_shift_data(threads, host_buffer, device_buffers, vector_length, bit_length, max_shift, filter_range):
    stats = np.zeros((len(device_buffers), max_shift))
    thread_list = list()
    for i in range(threads):
        thread_list.append(Process(target=gen_shift_data, args=(host_buffer,device_buffers[i],i,vector_length,bit_length,max_shift,5, stats,)))

    for j in range(threads):
        thread_list[j].start()

    for k in range(threads):
        thread_list[k].join()

    return stats

def threaded_shift_resist(threads, host_file, device_name, vector_length, bit_length, max_shift, harm_range):
    stats = np.zeros(harm_range)
    thread_list = list()
    for i in range(threads):
        thread_list.append(threading.Thread(target=gen_shift_resistance_by_harm, args=(host_file,device_name,vector_length,bit_length,max_shift,(harm_range//threads)*i,(harm_range//threads)*(i+1),stats,)))

    for j in range(threads):
        thread_list[j].start()

    for k in range(threads):
        thread_list[k].join()

    return stats

def drop_off_sample(stats):
    sample = len(stats)
    for i in range(len(stats)):
        if stats[0,i] < 80:
            sample = i
    return sample

def gen_shift_resistance_by_harm(host_file, device_name, vector_length, bit_length, max_shift, harm_range_start, harm_range_end, stats):
    device_file_names = [device_name]
    for harm in range(harm_range_start, harm_range_end):
        shift_stats = gen_shift_data(host_file, device_file_names, vector_length, bit_length, max_shift, harm)
        stats[harm] = drop_off_sample(shift_stats)
    print("done")

def graph(data, label_names):
    x = range(len(data[0,:]))
    data_plots = len(data[:,0])
    for i in range(data_plots):
        plt.plot(x, data[i], label=label_names[i])
    plt.legend()
    plt.xlabel("Sample Shift")
    plt.ylabel("Agreement Rate")
    plt.show()
        

if __name__ == "__main__":
    # Parameters
    #directory = "./data/doyle_and_condo/10msa/"
    #host_file = directory + "Doyle_basement_10msa_f32.bin"
    #device_file = directory + "ML_lab_10msa_f32.bin"
    #res_data = threaded_shift_resist(16, host_file, device_file, 2000, 64, 300, 64)  
    #graph(res_data, 64, 0)

    file = "./ml_hallway_5msa_ds4.csv"
    channels = 3
    sample_len = 200000
    buffers = np.zeros((channels,sample_len))
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for j in range(sample_len):
            row = next(reader)
            for i in range(1,channels+1):
                buffers[i-1,j] = float(row[i])
    print(buffers)
    stats = threaded_gen_shift_data(3, buffers[0], buffers, 2000, 64, 15000, 200) 
    label_names = ["Third Floor 1", "Third Floor 2", "ML Hallway"]
    graph(stats, label_names)

   # host_file = directory + "Doyle_basement_10msa_f32.bin"
   # device_file = [directory + "ML_lab_10msa_f32.bin"]
   #shift_stats = gen_shift_data(host_file, device_file, 2000, 64, 300, 5)
   # graph(shift_stats, 0)

