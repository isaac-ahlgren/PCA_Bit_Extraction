from bit_extract import tr_bit_extract
import numpy as np
import matplotlib.pyplot as plt
import threading

def compare_bits(bits1, bits2, bit_length):
    agreed = 0
    for i in range(bit_length):
        if bits1[i] == bits2[i]:
            agreed += 1
    return (agreed/bit_length)*100

def gen_shift_data(host_file, device_file_names, vector_length, bit_length, max_shift, filter_range):
    
    stats = np.zeros((len(device_file_names), max_shift))
    for i in range(len(device_file_names)):
        device_file = device_file_names[i]
        for shift in range(max_shift):
            host_samples = np.fromfile(host_file, dtype=np.float32, count=(vector_length*bit_length))
            device_samples = np.fromfile(device_file, dtype=np.float32, count=(vector_length*bit_length), offset=(shift*4))
            print(host_samples)
            print(device_samples)
            host_bits = tr_bit_extract(host_samples, bit_length, filter_range)
            device_bits = tr_bit_extract(device_samples, bit_length, filter_range)
            agreement_rate = compare_bits(host_bits, device_bits, bit_length)
            stats[i,shift] = agreement_rate
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
        plt.plot(x, data[i])
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

    directory = "./data/doyle_and_condo/10msa/"
    host_file = directory + "Doyle_basement_10msa_f32.bin"
    device_file = [directory + "ML_lab_10msa_f32.bin"]
    shift_stats = gen_shift_data(host_file, device_file, 2000, 64, 300, 5);
    graph(shift_stats, 0)

