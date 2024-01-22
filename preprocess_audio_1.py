
# coding: utf-8

# In[1]:


import numpy as np
import wave
import librosa
import scipy.io.wavfile
from scipy.fftpack import dct
from pydub import AudioSegment
from pydub.silence import split_on_silence
import matplotlib.pyplot as plt
import madmom
import scipy.signal
import os
from scipy.fftpack import fft, ifft,fftshift,fftfreq
from scipy.signal import butter, lfilter, freqz
import IPython.display 
get_ipython().run_line_magic('matplotlib', 'inline')


def resize_audio(signal,sample_rate,time=500):
    #convert it into 5 sec i.e. if short:pad   else trim
    if((len(signal)/sample_rate) < time):
        no_pads=time *sample_rate -len(signal)
        pads=np.zeros(no_pads)
        signal=np.append(signal,pads)
        
    else:
        signal=signal[0:int(time*sample_rate)]
    return signal    
#signal, sample_rate = madmom.audio.signal.load_wave_file(audio_file_name)
def emphasize_signal(signal,pre_emphasis = 0.97):
    return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

#signal=emphasize_signal(signal)
#audio_file_name="1.wav"
#signal, sample_rate = librosa.load(audio_file_name, res_type='kaiser_fast')

def create_frames(emphasized_signal,size=2048,sample_rate=8000):
    #Framing the audio signal
    frame_size=size/sample_rate #24ms                        128=sample_rate*time    thus time=128/sample_rate
    frame_overlap=frame_size/2 #15ms overlap( ~50%)
    frame_length=int(round(frame_size*sample_rate))
    stride_length=int(round((frame_overlap)*sample_rate))
    # Pad the emphasized_signal with zeros in end corresponding to frame size
    signal_length=len(emphasized_signal)
    print("signal_length:"+str(signal_length)+"  frame_length:"+str(frame_length)+"  stride_length:"+str(stride_length))

    no_strides=np.ceil((signal_length/stride_length))
    print("no_strides:"+str(no_strides))
    no_pads=int(abs(signal_length-no_strides*stride_length))
    print("no_pads:"+str(no_pads))
    z=np.zeros(no_pads)
    print(np.shape(z))
    emphasized_signal=np.append(emphasized_signal,z)
    no_frames=int(len(emphasized_signal))/(stride_length)
    print("no_frames:"+str(no_frames))
    print(np.shape(emphasized_signal))

    frames=[]
    counter=0
    for i in range(0,len(signal)):
        #frames[counter]=signal[i:(i+frame_length)]
        frames.insert(counter, signal[i:(i+frame_length)])
        i+=stride_length
        counter+=1
    
    return frames


path=''
def create_chunks(sound):
    #for i in sounds:
    sound_file = AudioSegment.from_wav(sound)
    print("Average dBFS silence :"+str(sound_file.dBFS))
    avg_silence_threshold=sound_file.dBFS
    audio_chunks = split_on_silence(sound_file, min_silence_len=500,silence_thresh=avg_silence_threshold-2)
    for j, chunk in enumerate(audio_chunks):
        out_file = "chunk{0}.wav".format(j)
        print("exporting", out_file)
        chunk.export(path+out_file, format="wav")
    print("Done Exporting")
    

intrusion_sounds=os.listdir("Dataset/1_intrusion")

#get the audio

audio_file_name="atm.wav"
signal, sample_rate = librosa.load(audio_file_name, res_type='kaiser_fast')
print("Length of signal before resizing: "+str(len(signal)/sample_rate))
   
signal=resize_audio(signal,sample_rate)   
print("Length of signal after resizing: "+str(len(signal)/sample_rate))

#Signal pre-emphasis
#signal, sample_rate = librosa.load(audio_file_name, res_type='kaiser_fast')
plt.subplot(2,2,1)
plt.plot(signal)


emphasized_signal=emphasize_signal(signal)
plt.subplot(2,2,2)
plt.plot(signal)
print("Length of signal after emphasis: "+str(len(emphasized_signal)/sample_rate))


frames=create_frames(emphasized_signal,2048,8000)
plt.subplot(2,2,3)
plt.plot(frames[550])


# Steps:-
# 
# 0)Resample,attenuate,normalize etc the signal to sample_rate=48000Hz
# 
# 1)Divide the audio into frames with han windows and 
# 
# 2)then perform noise reduction ,
# 
# 3)then combine the signal again and
# 
# 4)divide it into sound chunks for further classification

frames=np.vstack(frames).astype(None)


print(len(emphasized_signal)/sample_rate)
scipy.io.wavfile.write("resized_signal.wav",sample_rate,signal.astype('int16'))
IPython.display.Audio("resized_signal.wav")


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(frame, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, frame)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(frame, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, frame)
    return y

def remove_noise(frames,sample_rate=48000,low_cutoff=3000,high_cutoff=800):
    denoised_frames=[]
    for i in range(0,len(frames)):
        #b,a=butter(5, normal_cutoff, btype='low', analog=False)
        #fft=scipy.fftpack.fft(frames[i])
        filtered_signal=butter_lowpass_filter(frames[i],low_cutoff,sample_rate,5)   #select frequencies below 3000Hz
        filtered_signal=butter_highpass_filter(filtered_signal,high_cutoff,sample_rate,5) #select frequencies above 800Hz
        frames[i]=filtered_signal
        
remove_noise(frames)        



#create_chunks(signal)


#feature extraction for each audio, get the mfcc,onsets(abrupt changes),zero_cross_rate,roll-off,energy,energy_entropy,
#Spectral centroid ,spectral flux,spectral entropy,spectral rolloff, harmonic ratio and pitch,MFCC filterbanks and MFC,
#Chroma features,spectograms

def extract_features(audio_file):
    combined_feature=[]
    signal, sample_rate = madmom.audio.signal.load_wave_file(audio_file)
    signal=madmom.audio.signal.normalize(signal)#normalize the signal
    y=signal.astype('float')
    sr=sample_rate
    
    #fft_freqs=madmom.audio.stft.fft_frequencies(num_fft_bins, sample_rate) #num_fft_bins=len(fft)/2
    mfcc=librosa.feature.mfcc(y=signal.astype('float'), sr=sample_rate, S=None, n_mfcc=30)
    combined_feature.append(mfcc)
    
    energy=madmom.audio.signal.energy(signal)
    combined_feature.append(energy)
    
    rms=madmom.audio.signal.root_mean_square(signal)
    combined_feature.append(rms)
    
    spl=madmom.audio.signal.sound_pressure_level(signal, p_ref=None) #sound pressure level
    combined_feature.append(spl)
    
    fs = madmom.audio.signal.FramedSignal(signal, frame_size=2048, hop_size=512)
    combined_feature.append(fs)
    
    stft = madmom.audio.stft.STFT(fs) #short-time_fourier transform
    
    spec = madmom.audio.spectrogram.Spectrogram(stft) #magnitudes of stft are used for MIR tasks
    combined_feature.append(spec)
    
    sf = madmom.features.onsets.spectral_flux(spec)  #spectral_flux
    combined_feature.append(sf)
    
    chroma_stft=librosa.feature.chroma_stft(y=signal.astype('float'), sr=sample_rate)
    combined_feature.append(chroma_stft)
    
    
    mel1 = librosa.feature.melspectrogram(y=signal.astype('float'), sr=sample_rate)
    combined_feature.append(mel1)
    
    rmse=librosa.feature.rmse(y=y)
    combined_feature.append(rmse)
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    combined_feature.append(centroid)
    
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    combined_feature.append(spec_bw)
    
    flatness = librosa.feature.spectral_flatness(y=y)
    combined_feature.append(flatness)
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    combined_feature.append(rolloff)
    
    return combined_feature


    

extract_features("atm.wav")



