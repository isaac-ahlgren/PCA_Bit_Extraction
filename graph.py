from bit_extract import tr_bit_extract
import numpy as np
import matplotlib.pyplot as plt

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
            host_samples = np.fromfile(host_file, dtype=np.float64, count=(vector_length*bit_length))
            device_samples = np.fromfile(device_file, dtype=np.float64, count=(vector_length*bit_length), offset=(shift*8))
            host_bits = tr_bit_extract(host_samples, bit_length, filter_range)
            device_bits = tr_bit_extract(device_samples, bit_length, filter_range)
            agreement_rate = compare_bits(host_bits, device_bits, bit_length)
            stats[i,shift] = agreement_rate
    return stats

def graph(data, data_plots, data_points, label_names):
    x = range(data_points)
    for i in range(data_plots):
    	plt.plot(x, data[i])
    plt.show()
        

if __name__ == "__main__":
    # Parameters
    directory = "./data/electricity_data/"
    host_file = directory + "new_case1_50_ch1"
    device_file_names = [directory + "new_case1_50_ch1", directory + "IC_50_ch1", directory + "Cuneo_50_ch1", directory + "Doyle_Later_ch1"]
    shift_data = gen_shift_data(host_file, device_file_names, 2000, 64, 500, 2)
    graph(shift_data, len(device_file_names), 500, 0)
