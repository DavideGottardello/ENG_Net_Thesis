import numpy as np
import pywt
from scipy.signal import decimate
import time as time

# All the function that gives a feature must return a number

def wavelet_decomposition(signal):
    
    wavelet='sym7'
    level=3
    coeffs  = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

def cA3(x):
    cA3, cD3, cD2, cD1  = wavelet_decomposition(x)
    return MeanEnergy(cA3)

def cD3(x):
    cA3, cD3, cD2, cD1  = wavelet_decomposition(x)
    return MeanEnergy(cD3)

def cD2(x):
    cA3, cD3, cD2, cD1  = wavelet_decomposition(x)
    return MeanEnergy(cD2)

def cD1(x):
    cA3, cD3, cD2, cD1  = wavelet_decomposition(x)
    return MeanEnergy(cD1)

def DFTEnergy(x):
    # Compute the FFT of the signal
    fft_result = np.fft.fft(x)
    energy = np.sum(np.abs(fft_result)**2)
    return energy

def Wavelength(x):

    x = np.array(x)
    N = len(x)
    
    if N < 2:
        raise ValueError("Input array must have at least two elements.")

    WL = np.sum(np.diff(x)) / N

    return WL

def MeanEnergy(X):
    mean = np.mean(np.mean(X**2))
    return mean;

def Variance(X):
    return (np.std(X))**2

def HjorthMobility(X):
    
    # First derivative
    x0 = X.flatten()
    x1 = np.diff(np.concatenate([[0], x0]))
    
    # Standard deviation
    sd0 = np.std(x0)
    sd1 = np.std(x1)
    
    # Mobility
    HM = sd1 / sd0
    
    return HM

def HjorthComplexity(X):
    
    # First & second derivative
    x0 = X.flatten()
    x1 = np.diff(np.concatenate([[0], x0]))
    x2 = np.diff(np.concatenate([[0], x1]))
    
    # Standard deviation of first & second derivative
    sd0 = np.std(x0)
    sd1 = np.std(x1)
    sd2 = np.std(x2)
    
    # Complexity
    HC = (sd2 / sd1) / (sd1 / sd0)
    
    return HC



def fun_stelline(signals, timeon, timeoff, n):
    fs = 30000

    # Calculate variable 'intr'
    if n == 1:
        intr = 50
    elif n == 3:
        intr = 16
    elif n == 6:
        intr = 8
    else:
        raise ValueError('Variable n not accepted')

    fs = fs / n

    # Downsample timeon and timeoff
    timeon_dws = np.round(timeon / n).astype(int)
    timeoff_dws = np.round(timeoff / n).astype(int)

    template = np.zeros((16, intr * 2 + 1))
    matr_16_dec_values = np.zeros((16, 90000))
    matr_16_intorni = np.zeros((16, 2000))
    matr_16_vett_completo = np.zeros((16, 3000))

    # Loop for each electrode
    for chan in range(16):
        xtot = np.arange(0, len(signals[chan]) / fs, 1 / fs)
        vet0 = signals[chan]
        vet = decimate(vet0, n)

        # Calculate std mean between std of individual time on
        i = 1
        timeon_vectportions = np.zeros(timeoff-timeon)

        for k in range(len(timeon_dws)):
            vett_on = vet[timeon_dws[i - 1]:timeoff_dws[i]]
            timeon_vectportions[k, :len(vett_on)] = vett_on
            i += 1

        std_matr = np.std(timeon_vectportions, axis=1)
        std_mean = np.mean(std_matr)

        thr_inf = (4 * np.median(np.abs(vet))) / 0.6745
        thr_sup = 6 * std_mean

        a1 = np.where((vet > thr_inf) & (vet < thr_sup))[0]

        vett_bordi = np.ones(len(a1))
        nsample_finestra = np.round(fs * 0.0033).astype(int)
        nsample_destra = np.round(nsample_finestra / 2).astype(int)
        nsample_sinistra = nsample_destra - 1

        for c in range(len(a1)):
            if ((a1[c] - nsample_sinistra < 0) or (a1[c] + nsample_destra > len(vet))):
                vett_bordi[c] = 0

        a = a1[vett_bordi == 1]

        vett_spike = np.ones(len(a))

        for c in range(len(a)):
            left = a[c] - nsample_sinistra
            dex = a[c] + nsample_destra
            vett_porzione = vet[left:dex]

            for indice in range(len(vett_porzione)):
                if vett_porzione[indice] > thr_sup:
                    vett_spike[c] = 0

        stelline_rimaste = a[vett_spike == 1]

        vett_spike_2 = np.zeros(len(stelline_rimaste))

        for z in range(len(stelline_rimaste)):
            left2 = stelline_rimaste[z] - nsample_sinistra
            dex2 = stelline_rimaste[z] + nsample_destra
            stellineconsiderate = stelline_rimaste[
                (stelline_rimaste > left2) & (stelline_rimaste < dex2)
            ]
            vettporzione2 = vet[stellineconsiderate]

            if np.max(vettporzione2) == vet[stelline_rimaste[z]]:
                vett_spike_2[z] = 1
            else:
                vett_spike_2[z] = 0

        stelline_finali = stelline_rimaste[vett_spike_2 == 1]

        matr_16_indicifinali = np.zeros((16, len(stelline_finali)))
        matr_16_indicifinali[chan, :len(stelline_finali)] = stelline_finali

        a = vet[stelline_finali[0] - intr : stelline_finali[0] + intr + 1]
        for j in range(1, len(stelline_finali)):
            a = a + vet[stelline_finali[j] - intr : stelline_finali[j] + intr + 1]

        template[chan, :] = a / len(stelline_finali)

    return template



def compute_feature_for_channel(channel_data, feature_name):
    # Use globals() to get the function by name
    feature_function = globals().get(f'{feature_name}')
    
    # Check if the feature function exists
    if feature_function is not None and callable(feature_function):
        
        # Call the feature function with channel data
        start_time = time.time()
        result = feature_function(channel_data)
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = (end_time - start_time)*1000

        return result, elapsed_time
       
    else:
        print(f"Feature function {feature_name} does not exist.")
        return None

def compute_all_features_per_sample(x_samp_i, feature_names):
    num_channels, num_points = x_samp_i.shape
    all_features =[]
    all_times =[]
    for feature_name in feature_names:
        
        for channel in range(num_channels):

            # Select data for the current channel
            channel_data = x_samp_i[channel, :]
            channel_feature, elapsed_time = compute_feature_for_channel(channel_data, feature_name)
            all_features.append(channel_feature)
            all_times.append(elapsed_time)
            
    return all_features, all_times


def compute_all_features_for_all_data(x_samp, feature_names):
    
    all_sample_features = []
    all_feature_names=[]
    all_sample_times = []

    num_samples, num_channels, num_points = x_samp.shape
    
    for feature_name in feature_names:        
        for channel in range(num_channels):
            
            # Create a unique name for each channel's feature
            channel_feature_name = f"{feature_name}_{channel + 1}"
             # Append the channel feature name to the list
            all_feature_names.append(channel_feature_name)

    for x_samp_i in range(x_samp.shape[0]):
        
        x_sample_i = x_samp[x_samp_i, :, :]
    
        all_features, all_times = compute_all_features_per_sample(x_sample_i, feature_names)
        
        all_sample_features.append(all_features)
        all_sample_times.append(all_times)

        
    return all_sample_features, all_feature_names, np.mean(all_sample_times, axis=0)













