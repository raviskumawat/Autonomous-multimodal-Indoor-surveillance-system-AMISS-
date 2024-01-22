from python_speech_features import mfcc,delta,logfbank
import scipy.io.wavfile as wav

sample_rate,samples=wav.read('audio.wav')
mfccfeatures=mfcc(samples,sample_rate)
dmfcc=delta(mfccfeatures,2)
fbankfeature=logfbank(samples,sample_rate)

print(fbankfeature)