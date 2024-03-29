# Autonomous-multimodal-Indoor-surveillance-system-AMISS-
Autonomous multimodal surveillance system with modules consisting of audio and video analysis which can work in real-time on low hardware resources: Test System: Intel i5, Nvidia GTX 1050.



# Video Models
After pre-processing and converting into suitable representations, the following tasks
are performed:



# Video-flow
![Flow-Diagram for Video Module](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/VIDEO_final.png)



### Motion Detection: 
Start Video analysis on motion detection
![Detect motion and track objects](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/motion_detect.png)



### Object detection: 
Detect objects present
![Detect objects present](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/object_detect.png)

![](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/Object%20Detection.jpg)



### Multiface tracking, Face recognition and facial expression classifier: 
Check whether the person is known criminal
![Detect faces>>>>Recognise face and emotion](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/face_emo_rec.png)
![](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/Multiface%20racking%2C%20Recognition%20and%20Expression%20classifier.png)



### Image Captioning: 
Identify the situation in the photo
![Caption the images](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/img_caption.png)


# Audio-flow
![Flow-Diagram for Audio Module](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/audio_flow.png)


# Audio Models
After pre-processing and converting audio into suitable representations, the following
tasks are performed:

### Audio classification: 
Classify sounds present
![Classification on EC50](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/Audio_classify_graph.png)

### Speech emotion classification: 
Identify emotion if human sound present
![Spectrogram Features](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/Audio%20emotion.png)
![Identify emotion if human sound present](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/emotion_audio.png)



# Text Models
### Threat sentence Classification:
Classify the captions generated from video
![Classify the captions generated from video](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/threat_classify.png)




# Combined Audio Model
![](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/audio_out.png)

# Combined Video Model
![](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/video_out_3.png)


# Final Output
![](https://github.com/raviskumawat/Autonomous-multimodal-Indoor-surveillance-system-AMISS-/blob/master/output%20imgs/Final%20Output.png)
