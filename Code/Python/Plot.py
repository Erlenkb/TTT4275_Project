import numpy as np
import matplotlib.pyplot as plt
import os

####### Font values #######
SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
###########################

######## Third octave settings ########
x_ticks_third_octave = [100, 200, 500, 1000, 2000, 5000]
x_ticks_third_octave_labels = ["100", "200", "500", "1k", "2k", "5k"]
y_ticks_freq_db = [0,5,10,40,70,100]

third_octave_center_frequencies = [100, 125, 160, 200, 250,
            315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
            5000] #, 6300, 8000, 10000, 12500, 16000, 20000]

third_octave_lower = [89.1, 112, 141, 178, 224, 282, 355, 447, 562, 708, 891, 1122, 1413, 1778, 2239, 2818, 3548, 4467, 5623]#, 7079, 8913, 11220, 14130, 17780,22390]

######################################

###### IR settings ###################
x_ticks_IR = np.linspace(0.0083,0.01,5)
x_ticks_IR_labels = [str(round(i*1000,5)) for i in x_ticks_IR]


length_time = 0.1
start_time = 0.007
fs = 48000

start = int(fs * start_time)
stop = int(fs * length_time)
#C:\Users\erlen\VS code\TTT4180\A0\Data\EXPOSED INSULATION\Freq
path_exposed = "C:/Users/erlen/VS code/TTT4180/A0/Data/EXPOSED INSULATION"
path_plywood = "C:/Users/erlen/VS code/TTT4180/A0/Data/PLYWOD FRONT"


#print(os.listdir(path_exposed))  

def _nextpow2(i):
    n = 1
    while n < i : n*=2
    return n

""" 
count = 0
folder = "C:/Users/erlen/Project_Diffraction/Code/Data/PLYWOD FRONT/Freq Meas/"
for file_name in os.listdir(folder):
    # Construct old file name
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + str(count) + "_S01_R01.etx"

    # Renaming the file
    os.rename(source, destination)
    count += 1
"""

def _Lp_from_third_oct(arr):
    """Return the sound pressure level from the third octave band array
    Args:
        arr (_type_): Third octave band array
    Returns:
        _type_: Lp
    """
    Lp = 0
    for i in arr: Lp += i 
    return 10*np.log10(Lp)

def _fft_signal(array):
    #N = 48000 #
    N = _nextpow2(len(array))
    array = np.pad(array, (0,_nextpow2(len(array))-len(array)),"constant")
    y = np.fft.fft(array, N)[0:int(N/2)]/N
    #y = _runningMeanFast(y,2)
    p_y = 20*np.log10(np.abs(y))
    f = 48000*np.arange((N/2))/N
    return f, p_y


def _getFFT(arr):
    sp = np.pad(arr, (0,_nextpow2(len(arr))-len(arr)),"constant")
    sp = np.fft.fft(sp, _nextpow2(len(arr)))[0:int(_nextpow2(len(arr))/2)]# / _nextpow2(len(arr))
    sp = np.trim_zeros(sp, trim="fb")
    freq = np.fft.fftfreq(n=len(sp), d=1/fs)
    #return freq, 20*np.log10(np.abs(sp) / (2*10**(-5)))
    return np.fft.fftshift(freq), 20*np.log10(np.abs(np.fft.fftshift(sp)))

def _third_Octave_bands(freq, arr, third_oct):
    third_octave_banks = []
    single_bank = []
    i = 0
    for it, x in enumerate(freq):
        if (x > third_oct[-1]) : break
        if (x >= third_oct[i] and x < third_oct[i+1]): 
            single_bank.append(round(arr[it],2))
        if (x >= third_oct[i+1]):
            third_octave_banks.append(single_bank)
            i += 1
            single_bank = []
            single_bank.append(round(arr[it],2))
    filtered_array = []
    for n in third_octave_banks : filtered_array.append(_Lp_from_third_oct(n))
    return filtered_array

REC_String = ";PROJECT=TEST_CATT_TFE4580\n\nRECEIVERS\n1 "

def _generate_source_pos(rad_src,rad_rec):
    num_it = 36
    with open("SRC.LOC","r") as SRC_FILE: SRC_DATA = SRC_FILE.readlines()
    
    SRC_pos_x = [rad_src*np.cos(np.deg2rad(-x*5)) for x in range(num_it)]
    SRC_pos_y = [rad_src*np.sin(np.deg2rad(-x*5)) for x in range(num_it)]
    
    REC_pos_x = [rad_src*np.cos(np.deg2rad(180-x*5)) for x in range(num_it)]
    REC_pos_y = [rad_src*np.sin(np.deg2rad(180-x*5)) for x in range(num_it)]
    file_name=""
    for i in range(num_it):
        SRC_file_name = "SRC/SRC"+str(i)+".LOC"
        SRC_file=open(SRC_file_name, "w")
        SRC_DATA[6] = " POS = " + str(round(SRC_pos_x[i],3)) + " " + str(round(SRC_pos_y[i],3))+ " 1.62\n"
        SRC_file.writelines(SRC_DATA)  
        
        REC_temp = str(round(REC_pos_x[i],3))+" "+str(round(REC_pos_y[i],3))+str(" 1.62")
        REC_file_name = "REC/REC"+str(i)+".LOC"
        REC_file=open(REC_file_name, "w")
        REC_file.write(str(REC_String+REC_temp))

#_generate_source_pos(1.3,1.66)



def _new_data():
    for i in range(6):
        start1 = int(fs * 0.0083)
        stop1 = int(fs * 0.01)
        path1 = "C:/Users/erlen/Project_Diffraction/Code/Data/New_Data"
        path2 = "C:/Users/erlen/Project_Diffraction/Code/Data/New_Data"
        file1 = "{0}/M002{1}_S01_R01.etx".format(path1,i)
        file2 = "{0}/M003{1}_S01_R01.etx".format(path2,i)

        IR1 = np.loadtxt(file1, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR2 = np.loadtxt(file2, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")

        freq1, fft1 = _getFFT(np.abs(IR1[start1:stop1,1]))
        freq2, fft2 = _getFFT(np.abs(IR2[start1:stop1,1]))
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)

        ax.plot(IR1[:,0],IR1[:,1], label="Covered")
        ax.plot(IR2[:,0],IR2[:,1], label="Exposed")
        ax.set_title("Degrees of rotation: {0}".format(5*i))
        ax.set_xlim(start1,0.01)
        #ax.set_xticks(x_ticks_IR)
        #ax.set_xticklabels(x_ticks_IR_labels)
        ax.set_ylabel("Magnitude")
        ax.set_xlabel("Time [ms]")
        ax.grid()
        ax.legend()

        ax1.semilogx(freq1, np.abs(fft1), label="Exposed insulation")
        ax1.semilogx(freq2, np.abs(fft2), label="Covered insulation")
        ax1.set_title("Degrees of rotation: {0}".format(5*i))
        ax1.set_xlim(100,4000)
        #ax1.set_xscale("log")
        #ax1.set_xticks(x_ticks_third_octave)
        #ax1.set_xticklabels(x_ticks_third_octave_labels)
        #ax1.grid(which="major")
        #ax1.grid(which="minor", linestyle=":")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Magnitude")
        ax1.legend()

        plt.show()




def _create_plot_IR_FFT():
    diff_lst_ply = []
    diff_lst_exp = []
    diff_lst_pos = []
    diff_lst_neg = []
    degrees = []
    for i in range(1,18):
        file_freq = "{0}/Freq/{1}_S01_R01.etx".format(path_exposed,i)
        file_exposed = "{0}/IR Meas/{1}_S01_R01.etx".format(path_exposed,i)
        file_covered = "{0}/IR Meas/{1}_S01_R01.etx".format(path_plywood,i)
        file_exposed_rev = "{0}/IR Meas/{1}_S01_R01.etx".format(path_exposed,36-i)
        file_covered_rev = "{0}/IR Meas/{1}_S01_R01.etx".format(path_plywood,36-i)
        "C:/Users/erlen/VS code/TTT4180/A0/Data/New_Data/M00{0}_S01_R01.etx"
        file_reference = "C:/Users/erlen/VS code/TTT4180/A0/Data/New_Data/M00{0}_S01_R01.etx".format(20)
        
        Freq_exp = np.loadtxt(file_reference, dtype=float, skiprows=24, max_rows=59500, delimiter="\t")
        
        
        IR_Covered = np.loadtxt(file_covered, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR_Exposed = np.loadtxt(file_exposed, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR_Covered_rev = np.loadtxt(file_covered_rev, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        IR_Exposed_rev = np.loadtxt(file_exposed_rev, dtype=float, skiprows=22, max_rows=int(fs*length_time), delimiter="\t")
        
        freq_val = 20*np.log10(np.abs(Freq_exp[:,1]+1j*Freq_exp[:,2]))
        freq = Freq_exp[:,0]
        
        fig1 = plt.figure(figsize=(10,7))
        plt.style.use('ggplot')
        ax2 = fig1.add_subplot(111)
        ax2.semilogx(freq, freq_val, label="Freq - {0} deg".format(5*i))
        ax2.set_title("Frequency response")
        ax2.set_xlim(100,4000)
        ax2.set_ylim(-20,25)
        ax2.set_xscale("log")
        ax2.set_xticks(x_ticks_third_octave)
        ax2.set_xticklabels(x_ticks_third_octave_labels)
        ax2.grid(which="major", color="dimgray")
        ax2.grid(which="minor", linestyle=":", color="dimgray")
        ax2.set_xlabel("Frequency [Hz]")
        ax2.set_ylabel("Magnitude [dB]")
        #ax2.legend(bbox_to_anchor=(1, 0.5), loc="center left")
        plt.close()
        #plt.show()
        
        freq_Exposed, fft_exposed = _getFFT(np.abs(IR_Exposed[start:stop,1]))
        freq_Covered, fft_Covered = _getFFT(np.abs(IR_Covered[start:stop,1]))
        freq_Exposed_rev, fft_exposed_rev = _getFFT(np.abs(IR_Exposed_rev[start:stop,1]))
        freq_Covered_rev, fft_Covered_rev = _getFFT(np.abs(IR_Covered_rev[start:stop,1]))

        third_oct_exposed = _third_Octave_bands(freq_Exposed, np.abs(fft_exposed), third_octave_lower)
        third_oct_covered = _third_Octave_bands(freq_Covered, np.abs(fft_Covered), third_octave_lower)

        fig = plt.figure(figsize=(10,7))
        plt.style.use('ggplot')
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)

        ax.plot(IR_Exposed[:,0],IR_Exposed[:,1], label="Exposed insulation: {0} deg".format(5*i))
        ax.plot(IR_Covered[:,0],IR_Covered[:,1], label="Covered insulation: {0} deg".format(5*i))
        ax.plot(IR_Exposed_rev[:,0],IR_Exposed_rev[:,1], label="Exposed insulation: {0} deg".format(180-5*i), linestyle="--" )
        ax.plot(IR_Covered_rev[:,0],IR_Covered_rev[:,1], label="Covered insulation: {0} deg".format(180-5*i), linestyle="--")
        
        #ax.set_title("Degrees of rotation: {0}".format(5*i))
        ax.set_title("Impulse response")
        ax.set_xlim(0.0084,0.01)
        ax.set_xticks(x_ticks_IR)
        ax.set_xticklabels(x_ticks_IR_labels)
        ax.set_ylabel("Magnitude [Pa]")
        ax.set_xlabel("Time [ms]")
        #ax.grid()
        ax.legend()

        ax1.semilogx(freq_Exposed, np.abs(fft_exposed), label="Exposed insulation: {0} deg".format(5*i))
        ax1.semilogx(freq_Exposed_rev, np.abs(fft_exposed_rev), linestyle="--", label="Exposed insulation: {0} deg".format(180-5*i))
        ax1.semilogx(freq_Covered, np.abs(fft_Covered), label="Covered insulation: {0} deg".format(5*i))
        ax1.semilogx(freq_Covered_rev, np.abs(fft_Covered_rev), linestyle="--",label="Covered insulation: {0} deg".format(180-5*i))
        ax1.semilogx(freq_Covered, (np.abs(fft_exposed)-np.abs(fft_exposed_rev)), label="Diff. Exposed")
        ax1.semilogx(freq_Covered, (np.abs(fft_Covered)-np.abs(fft_Covered_rev)), label="Diff. Covered")
        #ax1.set_title("Degrees of rotation for\n -Solid line: {0}\n-Dashed Line: {1}".format(5*i, 180-5*i))
        ax1.set_title("Frequency response")
        ax1.set_xlim(100,4000)
        ax1.set_ylim(-15,30)
        ax1.set_xscale("log")
        #ax1.set_yscale("log")
        #ax1.set_ylim(0,123)
        ax1.set_xticks(x_ticks_third_octave)
        ax1.set_xticklabels(x_ticks_third_octave_labels)
        #ax1.set_yticks(y_ticks_freq_db)
        ax1.grid(which="major", color="dimgray")
        ax1.grid(which="minor", linestyle=":", color="dimgray")
        ax1.set_xlabel("Frequency [Hz]")
        ax1.set_ylabel("Magnitude [dB]")
        ax1.legend()#bbox_to_anchor=(1, 0.5), loc="center left")
        plt.tight_layout()
        
        start_delta = int(len(freq_Exposed)/2) + int(300/ (freq_Exposed[1]-freq_Exposed[0]))
        stop_delta = int(len(freq_Exposed)/2) + int(5000 / (freq_Exposed[1]-freq_Exposed[0]))
        
        
        
        #Delta_ply = np.round(np.mean(np.abs((np.abs(fft_Covered[start_delta:stop_delta])-np.abs(fft_Covered_rev[start_delta:stop_delta]))/np.abs(fft_Covered[start_delta:stop_delta]))),3)*100
        #Delta_abs = np.round(np.mean(np.abs(np.abs(fft_exposed[start_delta:stop_delta])-np.abs(fft_exposed_rev[start_delta:stop_delta]))),3)
        
        Delta_ply = np.round(np.mean(np.abs((np.abs(fft_Covered[start_delta:stop_delta])-np.abs(fft_Covered_rev[start_delta:stop_delta]))))/np.mean(np.abs(fft_Covered[start_delta:stop_delta]))*100,3)
        Delta_abs = np.round(np.mean(np.abs((np.abs(fft_exposed[start_delta:stop_delta])-np.abs(fft_exposed_rev[start_delta:stop_delta]))))/np.mean(np.abs(fft_Covered[start_delta:stop_delta]))*100,3)
        
        Delta_meas_neg = np.round(np.mean(np.abs((np.abs(fft_Covered_rev[start_delta:stop_delta])-np.abs(fft_exposed_rev[start_delta:stop_delta]))))/np.mean(np.abs(fft_Covered_rev[start_delta:stop_delta]))*100,3)
        Delta_meas_pos = np.round(np.mean(np.abs((np.abs(fft_Covered[start_delta:stop_delta])-np.abs(fft_exposed[start_delta:stop_delta]))))/np.mean(np.abs(fft_Covered[start_delta:stop_delta]))*100,3)
        
       
        
        diff_lst_ply.append(Delta_ply)
        diff_lst_exp.append(Delta_abs)
        diff_lst_pos.append(Delta_meas_pos)
        diff_lst_neg.append(Delta_meas_neg)
        
        
        degrees.append(i*5)
        #print("Start val: {0} \t stop val:{1}".format(freq_Exposed[start_delta],freq_Exposed[stop_delta]))
        #print("Start val: {0} \t stop val:{1}".format(freq_Exposed[start_delta1],freq_Exposed[stop_delta1]))
        print("Delta_ply: {0:.0%}\t Delta_abs: {1:.0%}\t Deg: {2}".format(Delta_ply,Delta_abs,i*5))
        
        
        
        #fig.savefig("Pictures/Angle_Plot_Covered_Vs_Exposed_{0}_Deg.png".format(i*5))
        plt.close(fig)
        #plt.show()
        
        #print(Exposed_IR)
    
    degrees = list(reversed(degrees))
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(degrees,diff_lst_ply, label="Diff. Covered")
    ax.plot(degrees,diff_lst_exp, label="Diff. Exposed")
    
    ax.legend()
    ax.set_title("Difference in percentage between\n85-5 degrees and 95-175 degrees")
    ax.set_xlabel("Rotation from ref. @ 90 degrees [deg]")
    ax.set_ylabel("Difference [%]")
    fig.savefig("Pictures/Difference_Normal_VS_Reversed.png")
    plt.close()
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(degrees,diff_lst_pos, label="Diff. positive")
    ax.plot(degrees,diff_lst_neg, label="Diff. negative")
    ax.legend()
    ax.set_title("Difference in percentage between\nexposed and covered frontpanel")
    ax.set_xlabel("Rotation from ref. @ 90 degrees [deg]")
    ax.set_ylabel("Difference [%]")
    fig.savefig("Pictures/Difference_Covered_VS_Exposed.png")
    plt.close()

_create_plot_IR_FFT()