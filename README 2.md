# ATM-intrusion-detection
Problem Statement:
The model identifies an ongoing theft/robbery inside an ATM by analysing the sounds using machine learning approaches and send an alert message to the bank and police.


The project implements following functionalities:

i)Records audio continuously and analyse each 5-sec segment.    

ii)Remove the noise using FFT techniques.                  

iii)Train a GMM classifier model(scream, gunshots, crying etc) with ATM sounds.     

iv)Classify the 5-sec sound using prediction from the trained GMM model.       

v)If the sound is classified as intrusion take following steps:                      
              a)Start recording the video.                                 
              b)send an alert to bank with the live stream video of ATM and provide an option to inform Police.                     

## Ongoing improvements: 
Provide a web-UI implementation and use the video camera for Posture detection of people inside ATM.





















## NOTES/IDEAS for further improvement

:-Check out the "atm.wav" for sample noise and conversation in ATM (recorded by me and yash REAL DATA)


 1)fear  vs. neutral classification  for all people present in the room individually     (Emotion Analysis using sound)
           -surya:oye bhosdike chal paise nikal
            ravi:nikal raha hu bhai(clam)
																		  
 2)Load with all common all languages warning/threatning sentences and just build a simple classifier(chal pasise nikaal ,give me the money etc.)
    (keyword spotting) 
 
 3) Traffic horns,opening/closing doors, people talking in the background, train movement 
 
 4)The incoming signal is first classified as  normal (metro station
   environment) or abnormal (scream, gunshot or explosion) and in
   case it is decided to be abnormal the system proceeds into a second
   processing stage where the type of abnormality is identified
   
 5) make Two decision tree models  
     1:Extract all sound peaks and mark the label as 1 and other ATM sounds and normal conversations as 0
	 2:Extract all "situation words in various Indian languages" make them as 1 .
	 *** Give a threshold values like model requires at least 5 classifications in order to set the alarm.
	 *** Give very high weights to special sounds such as "Gun shots" etc.


1)Acoustic based intrusion detection system in Indian ATMs : The model uses two decision tree classifiers ,one classifies the ongoing action based on pitch, amplitude, shrillness,etc recorded by microphone and other tries to classifies based on common words used during attacks such as "please","gun" etc.


FFT:
* How to get the amount of each frequency in a signal?
* How audio signals are represented digitally?
* At one instance of time there are multiple frequencies or the whole audio signal can be created with n waves and thus n frequencies?
* How to extract and subtract noise from the ".wav"  ?
* Nyquist limit??????
* FTT gives what a constant repeating signal is composed of like "peeeeeeeiiiiiiii" but a sound is composed of many such signals at different times thus we need to sample the audio signal(maybe by windowing) and find the frequencies in that time period? Maybe we can have a Global time for which all frequencies are played i.e total-time-played vs frequency graph to extract the noise. 
* I can get the frequencies present in the silence part of the audios and make them very small in the FFT of the original wave and again combine the FFT to get a new de-noised sound.
*FFT+FFT=original wave?

*open source projects deep learning 

* create a formula based on slope and amplitude which lets pass certain frequencies only



## Readme2



Can we do pose estimation using image?




Find human sound or not .if human  then do speech recognition and  compare with language models else compare with thud,thum,glassbreaking,gunshots etc.


can we have dynamic sized input to neural network or  paded input i.e. sound waves of different lengths.

wrapper
approach, consists of evaluating a feature vec-
tor on the basis of classification results, obtained using that specific
subset of features


using the Figueiredo and Jain algorithm 


Shout detection

Baum-Welch algorithm

normal vs abnorma sound classification

J.L. Rouas, J. Louradour, and S. Ambellouis. Audio Events Detection in Public Transport Vehicle.

P.K. Atrey, N.C. Maddage, and M.S. Kankanhalli. Audio Based Event Detection for Multimedia Surveillance.




sudden burst of sound (aka onset)

Both gain and levels refer to the loudness of the audio

Attenuation:: how strongly the transmitted ultrasound amplitude decreases as a function of frequency

sound pressure level :::SPL is actually a ratio of the absolute, Sound Pressure and a reference level (usually the Threshold of Hearing, or the lowest intensity sound that can be heard by most people).


Framing:: If used in online real-time mode the parameters origin and num_frames should be set to ‘stream’ and 1, respectively.


We can send audio also along with the video



 class madmom.audio.signal.Stream(sample_rate=None, num_channels=None, dtype=<type 'numpy.float32'>, frame_size=2048, hop_size=441.0, fps=None, **kwargs)[source]

    A Stream handles live (i.e. online, real-time) audio input via PyAudio.

https://www.labri.fr/perso/nrougier/teaching/matplotlib/

http://scipy.github.io/devdocs/tutorial/fftpack.html

https://www.swharden.com/wp/2013-05-09-realtime-fft-audio-visualization-with-python/


http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

https://pypi.python.org/pypi/pocketsphinx

https://www.google.co.in/search?client=firefox-b-ab&dcr=0&biw=1366&bih=664&ei=4qKzWqXUKoXcvQS3oYewBQ&q=noise+reduction+from+audio+papers+top&oq=noise+reduction+from+audio+papers+top&gs_l=psy-ab.3..33i21k1j33i160k1.47080.55220.0.55446.18.18.0.0.0.0.314.2066.0j9j1j1.11.0....0...1c.1.64.psy-ab..10.8.1432...0j35i39k1j0i22i30k1j33i22i29i30k1.0.6YwIefgCJ1U

https://github.com/tyiannak/pyAudioAnalysis
http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144610
http://recognize-speech.com/preprocessing/harmonic-decomp

https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction

file:///D:/ATM%20intrusion%20detection/metro%20surveillance.pdf ::DONE

delta  energy  feature  is  that  it  is  
relatively  small  and  smooth  for  various  noise  background  
levels, but relatively large at abnormal acoustic events

Hamming window to frames


Two 8-state parallel left-right 
HMMs;  one  for  screamed  and  
one  for  non-screamed  are  
trained using forward-backward algorithm

Can provide for criminal sound recognition as well as an end product
https://www.geeksforgeeks.org/speech-recognition-in-python-using-google-speech-api/

The
incoming signal is first classified as  normal (metro station
environment) or abnormal (scream, gunshot or explosion) and in
case it is decided to be abnormal the system proceeds into a second
processing stage where the type of abnormality is identified. 


file:///D:/ATM%20intrusion%20detection/F04122034037.pdf

1500Hz-3.3 KHz

* pitch of every human being is unique(maximum coefficient in frequency zone)

*we can use complementary noise reduction i.e. encoded source in final project


for good frequency resolution , the length of signal for which to find fft should be in powers of 2.


most abundant frequency in a periodic signal


https://trainings.analyticsvidhya.com/dashboard

HMMs  can  be
efficiently trained using the Baum-Welch forward-backward training algorithm [4] or the Viterbi
algorithm [5]. As mentioned e

 fftfreq function (it returns negative frequencies instead of ones above the Nyquist

Decision tree-based state tying

file:///D:/ATM%20intrusion%20detection/6-162-1440572779132-135.pdf   ::DONE

perceptual features
, e.g.  loudness, sharpness

 the limits of the frequency range for
filtering the autocorrelation function have been fixed to 300
−
800
Hz: experimental results have shown that most of the energy of the
screams events is concentrated in this frequency range


we can provide a functionality of dynamic training i.e if the person choose to ignore then update the weights accordingly for wrong classification and add the example to the database.


Since we are working on 8k .wav file , we need to sample the input to 8k and into .wav file


After a preliminary segmentation step,  a
set of perceptual features such as MFCC (Mel-Frequency Cepstral
Coefficients) or PLP (Perceptual Linear Prediction) coefficients are
extracted from audio segments and used to perform a 3-levels clas-
sification.  First, the audio segment is classified either as noise or
non-noise;  second,  if it is not noise,  the segment is classified ei-
ther as speech or not speech; finally, if speech, it is classified as a
shout or not. The authors have tested this system using both GMMs
and Support Vector Machines (SVMs) as classifiers, showing that
in general GMMs provide higher precision.
 ::

https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/   :::DONE

https://www.kdnuggets.com/2016/09/urban-sound-classification-neural-networks-tensorflow.html/2    ::DONE

file:///D:/ATM%20intrusion%20detection/sound_detection.pdf   ::DONE important very     AN ABNORMAL SOUND DETECTION AN
D CLASSIFICATION SYSTEM FOR 
SURVEILLANCE APPLICATION
S 

https://www.safaribooksonline.com/library/view/elegant-scipy/9781491922927/ch04.html


active noise control  (ANC)   :- by  addition  of a  second  sound, specifically designed to cancel the existing one  S.  Chakrabarty, 
S.  Maitra
,  “Design  of  IIR  Digital  High  pass 
Butterworth Filter using Analog to Digital Mapping Technique”, 



Sampat and Vithalani
, 
2012
presented the de
-
noising 
of one dimensional signal using threshold is one of the 
major 
applications 
of 
wavelet 
transform. 
Determination of threshold type a
nd threshold value is 
one   of   the   important   tasks   in   threshold   based 
de
-
noising techniques.



Ramli 
et  al.,  2012
presented 
a  new  adaptive  filter 
whose coefficients are dynamically changing with an 
evolutionary    computation    algorithm    and    hence 
reducing    the    noise.    This    algorithm    gives    a 
re
lationship between the update rate and the minimum 
error  which  automatically  adjusts  the  update  rate. 
Adaptive Noise Cancellation is an alternative way  of 
cancelling noise present in a corrupted 
signal.



most abundant frequency in a periodic signal


from the full set of 47 features, we can build a feature vec-
tor of any dimension
l
,1
≤
l
≤
47. It is desirable to keep
l
small in
order to reduce the computational complexity of the feature extrac-
tion process and to limit the over-fitting produced by the increas-
ing number of parameters associated to features in the classification
model




https://notepad.pw/intrusionmodels

https://notepad.pw/raviATMlinks

https://notepad.pw/noise_removal

