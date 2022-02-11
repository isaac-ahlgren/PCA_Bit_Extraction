import os
from bit_extract import *
from graph import *

config_elec = { "channels": 3, "obs_vector_length": 2000, "bit_key_length": 64, "max_shift":5000, "filter_range":100, "sample_len": 200000 }


def parse_files(path):
    electric_data = {}
    audio_data = {}


    for root, dirs, files in os.walk(path,topdown=False):
        for fi in files:
            tmp_path = root+"/"+fi
            if ".csv" in tmp_path and "electricity" in tmp_path:
                electric_data[fi] = tmp_path
            elif ".csv" in fi and "audio_data" in tmp_path:
                audio_data[fi] = tmp_path 
    
    return audio_data,electric_data




def sample_csv(path, config_options):
    
    channels = config_options["channels"]
    obs_vector_length = config_options["obs_vector_length"]
    bit_key_length = config_options["bit_key_length"]
    max_shift = config_options["max_shift"]
    filter_range = config_options["filter_range"]
    sample_len = config_options["sample_len"]


    buffers = np.zeros((channels,sample_len))
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for j in range(sample_len):
            row = next(reader)

            for i in range(1,channels+1):
                buffers[i-1,j] = float(row[i])
    
    for i in range(buffers.shape[0]):
        stats = gen_shift_data_jack(buffers[0], buffers[i], obs_vector_length, bit_key_length, max_shift, filter_range, i)
        #Delete this if you want to do more devices.
        exit()
    
    print(stats)

if __name__ == "__main__":
    audio,elect = parse_files("../")
    for key in elect.keys():
        sample_csv(elect[key], config_elec)
        exit()
