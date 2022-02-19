import os
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from bit_extract import *
from graph import *
from l1_pca_file import *


parser = argparse.ArgumentParser()
parser.add_argument("generate_bit_streams", help="Supposed to be a 1 or 0. 1 If you would like to generate a bit stream or 0 if you do not want to.")
parser.add_argument("run_experiments", help="Supposed to be a 1 or 0. 1 If you would like to run experiments or 0 if you do not want to.")


config_elec = { "channels": 3, "obs_vector_length": 2000, "bit_key_length": 64, "max_shift":5000, "filter_range":100, "sample_len": 200000 }


experiment_config = { "channel_1_shift" : 98, "channel_2_shift" : 98, "channel_3_shift" : 98, "plot_raw_sample" : False,  "plot_data_matrix": True, "plot_fft" : False, "plot_covariance" : False, 
"plot_eigen_vectors" : True, "display_eigen_value" : False, "plot_proj_data" : False, "use_l1_pca":False, "fix_direction": True, "vector_num":64, "filter_range": 100 }





def idea_bin(data1,data2,data3):
    base_sinisoid = lambda t: 169.8*np.sin(377*t)
    X = np.arange(len(data1))
    Y = base_sinisoid(X)
    print(Y)
    YF = np.fft.fft(Y)
    vector_num = 64
    data_matrix1 = np.array(np.split(data1, vector_num)).T
    data_matrix2 = np.array(np.split(data2, vector_num)).T
    data_matrix3 = np.array(np.split(data3, vector_num)).T

    vlen = vector_num
    
    fft_data_matrix1 = np.abs(np.fft.fft(data_matrix1))
    fft_data_matrix2 = np.abs(np.fft.fft(data_matrix2))
    fft_data_matrix3 = np.abs(np.fft.fft(data_matrix3))

    cov_matrix1 = np.cov(fft_data_matrix1)
    cov_matrix2 = np.cov(fft_data_matrix2)
    cov_matrix3 = np.cov(fft_data_matrix3)


    output1 = la.ssyevx(cov_matrix1.T, range='I', il=vlen, iu=vlen)
    output2 = la.ssyevx(cov_matrix2.T, range='I', il=vlen, iu=vlen)
    output3 = la.ssyevx(cov_matrix3.T, range='I', il=vlen, iu=vlen)
    
    eig_vec1 = np.array(output1[1])[0]
    eig_vec2 = np.array(output2[1])[0]
    eig_vec3 = np.array(output3[1])[0]


    print(eig_vec1.shape)



    import pywt

    c = base_sinisoid(2)
    dec_lo, dec_hi, rec_lo, rec_hi = [c, c], [-c, c], [c, c], [c, -c]
    filter_bank = [dec_lo, dec_hi, rec_lo, rec_hi]
    myWavelet = pywt.Wavelet(name="deez wavelets", filter_bank=filter_bank)

    print (pywt.wavelist('sym'))

    coeffs1 = pywt.dwt(eig_vec1, 'sym3')
    coeffs2 = pywt.dwt(eig_vec2, 'sym3')
    coeffs3 = pywt.dwt(eig_vec3, 'sym3')
    print(coeffs1)
    print(coeffs2)
    print(coeffs3)

    # (cA1, cD1) = pywt.dwt(cA1-cAY, 'db2')
    # (cA2, cD2) = pywt.dwt(cA2-cAY, 'db2')
    # (cA3, cD3) = pywt.dwt(cA3-cAY, 'db2')

    cB1 = ""
    cB2 = ""
    cB3 = ""
    
    
    # cB1 = np.abs(coeffs1-coeffs2)
    # cB2 = np.abs(coeffs1-coeffs3)
    # cB3 = np.abs(coeffs2-coeffs3)

    # print(cB1.mean())
    # print(cB2.mean())

    #plot_data([cB1,cB2,cB3], 1)
    
    
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

#data =  is a numpy array
''' 
dims = describes how many dimensions the data is supposed to be.
       If the data has two dimensions then the x axis will be one of the dimensions in the given data set.
'''

def plot_data(data, dims, fft=False, sample_len=0,sample_spacing=0, heat_map=False):
    d1 = data[0]
    d2 = data[1]
    d3 = data[2]


    figure, axis = plt.subplots(1,3)
    
    if dims == 1:
        X = [i for i in range(len(d1))]
        axis[0].plot(X,d1)
        axis[0].set_title("Device 1")
        axis[1].plot(X,d2)
        axis[1].set_title("Device 2")
        axis[2].plot(X,d3)
        axis[2].set_title("Device 3")
        plt.show()
        plt.clf()

    if fft:
        n = len(d1) # length of the signal
        k = np.arange(n)
        T = n//sample_len
        frq = k/T # two sides frequency range
        #frq = frq[range(n//2)] # one side frequency range

        Y1 = np.fft.fft(d1)/n # fft computing and normalization
        Y1 = Y1[range(n)]
        Y2 = np.fft.fft(d2)/n # fft computing and normalization
        Y2 = Y2[range(n)]
        Y3 = np.fft.fft(d3)/n # fft computing and normalization
        Y3 = Y3[range(n)]

        axis[0].plot(frq,abs(Y1))
        axis[0].set_title("Device 1")
        axis[1].plot(frq,abs(Y2))
        axis[1].set_title("Device 2")
        axis[2].plot(frq,abs(Y3))
        axis[2].set_title("Device 3")
        plt.show()
        plt.clf()

    if heat_map:
        sns.heatmap(np.abs(d1-d2),ax=axis[0],vmin=0, vmax=1) 
        axis[0].set_title("Covariance 1 - Covariance 2")
        sns.heatmap(np.abs(d1-d3),ax=axis[1],vmin=0, vmax=1)
        axis[1].set_title("Covariance 1 - Covariance 3")
        sns.heatmap(np.abs(d2-d3),ax=axis[2],vmin=0, vmax=1)
        axis[2].set_title("Covariance 2 - Covariance 3")
        plt.show()
        plt.clf()

    if eigen:
        sns.heatmap(np.abs(d1-d2),ax=axis[0],vmin=0, vmax=1) 
        axis[0].set_title("Covariance 1 - Covariance 2")
        sns.heatmap(np.abs(d1-d3),ax=axis[1],vmin=0, vmax=1)
        axis[1].set_title("Covariance 1 - Covariance 3")
        sns.heatmap(np.abs(d2-d3),ax=axis[2],vmin=0, vmax=1)
        axis[2].set_title("Covariance 2 - Covariance 3")
        plt.show()
        plt.clf()

def tr_bit_extract_debugging(config):


    vector_num = config["vector_num"]
    filter_range = config["filter_range"]
    data1 = config["dev1_data"]
    data2 = config["dev2_data"]
    data3 = config["dev3_data"]
    use_l1 = config["use_l1_pca"]
    fix_d = config["fix_direction"]


    if config["plot_raw_sample"]:
        plot_data([data1,data2,data3],1)

    idea_bin(data1,data2,data3)
    return

    data_matrix1 = np.array(np.split(data1, vector_num))
    data_matrix2 = np.array(np.split(data2, vector_num))
    data_matrix3 = np.array(np.split(data3, vector_num))

    
    vlen = len(data1) // vector_num
    
    fft_data_matrix1 = np.abs(np.fft.fft(data_matrix1))
    fft_data_matrix2 = np.abs(np.fft.fft(data_matrix2))
    fft_data_matrix3 = np.abs(np.fft.fft(data_matrix3))

    if config["plot_fft"]:
        plot_data([data1,data2,data3],2,fft=True,sample_len=2000,sample_spacing=1/vlen)

    cov_matrix1 = np.cov(fft_data_matrix1.T)
    cov_matrix2 = np.cov(fft_data_matrix2.T)
    cov_matrix3 = np.cov(fft_data_matrix3.T)

    if config["plot_covariance"]:
        plot_data([cov_matrix1,cov_matrix2,cov_matrix3],2,heat_map=True,sample_len=2000,sample_spacing=1/vlen)


    if not use_l1:
        output1 = la.ssyevx(cov_matrix1, range='I', il=vlen, iu=vlen)
        output2 = la.ssyevx(cov_matrix2, range='I', il=vlen, iu=vlen)
        output3 = la.ssyevx(cov_matrix3, range='I', il=vlen, iu=vlen)
        
        eig_vec1 = np.array(output1[1])
        eig_vec2 = np.array(output2[1])
        eig_vec3 = np.array(output3[1])
    
    else:
        rank_r = 1	    	# Number of L1-norm principal components.
        num_init = 50 		# Number of initializations.
        print_flag = True	# Print decomposition statistics (True/False).

        output1 = l1pca_sbfk(cov_matrix1, rank_r, num_init, print_flag)
        output2 = l1pca_sbfk(cov_matrix2, rank_r, num_init, print_flag)
        output3 = l1pca_sbfk(cov_matrix3, rank_r, num_init, print_flag)

        eig_vec1 = np.array(output1[1])
        eig_vec2 = np.array(output2[1])
        eig_vec3 = np.array(output3[1])

    if config["plot_eigen_vectors"]:
        plot_data([eig_vec1,eig_vec2,eig_vec3],1)

    if fix_d:
        eig_vec1 = fix_direction(eig_vec1)
        eig_vec2 = fix_direction(eig_vec2)
        eig_vec3 = fix_direction(eig_vec3)

    proj_data1 = (eig_vec1.T).dot(fft_data_matrix1.T)
    proj_data2 = (eig_vec2.T).dot(fft_data_matrix2.T)
    proj_data3 = (eig_vec3.T).dot(fft_data_matrix3.T)

    proj_data1 = proj_data1[0]
    proj_data2 = proj_data2[0]
    proj_data3 = proj_data3[0]

    # return gen_bits(proj_data)



def main(path, config_options,device,folder_name,generate_all_bit_streams,run_experimentation,experiment_config=experiment_config):
    
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

    
    if generate_all_bit_streams == 1:
        for i in range(buffers.shape[0]):
            stats = gen_shift_data_jack(buffers[0], buffers[i], obs_vector_length, bit_key_length, max_shift, filter_range, f"{device}_{i}",folder_name)

    elif run_experimentation == 1:
        dev1 = buffers[0]
        dev2 = buffers[1]
        dev3 = buffers[2]  
        shift1 = experiment_config["channel_1_shift"]
        shift2 = experiment_config["channel_2_shift"]
        shift3 = experiment_config["channel_3_shift"]
        experiment_config["dev1_data"] = dev1[shift1:(shift1 + obs_vector_length*bit_key_length)]
        experiment_config["dev2_data"] = dev2[shift2:(shift2 + obs_vector_length*bit_key_length)]
        experiment_config["dev3_data"] = dev3[shift3:(shift3 + obs_vector_length*bit_key_length)]
        tr_bit_extract_debugging(experiment_config)
    
    
    # print(stats)

if __name__ == "__main__":
    audio,elect = parse_files("../")
    args = parser.parse_args()
    gen_bits = int(args.generate_bit_streams)
    run_exps = int(args.run_experiments)

    for key in elect.keys():
        device = elect[key].split("_")[2]
        folder_name = elect[key].split("/")[-1].replace(".csv","")
        if ".csv" in device:
            device.replace(".csv","")
        
        if "ds20" in elect[key]:
            print(run_exps)
            main(elect[key], config_elec, device, folder_name,gen_bits,run_exps)
        
