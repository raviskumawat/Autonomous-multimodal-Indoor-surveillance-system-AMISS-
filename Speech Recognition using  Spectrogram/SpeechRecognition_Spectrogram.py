import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
#import IPython.display as play
#%matplotlib inline

sample_rate,samples=wav.read('audio.wav')
freq,times,spectogram=signal.spectrogram(samples,sample_rate)
'''plt.imshow(spectogram,aspect='auto')
plt.imshow()'''

plt.plot(spectogram)
plt.xlabel('Time (sec)')
plt.ylabel('Freq (Kz)')
plt.show()

#play.Audio('audio.wav')

