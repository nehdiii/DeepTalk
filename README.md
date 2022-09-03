# DeepTalk

## 1) Project description 

This Project is done during My summer internship in Talan Tunisie some of company work is in metaverse and building meta worlds so i wanted to use what i learned during my trip in AI and Data Science and so the idea of the project is to build a chat bot to help users in metaverses but, the special thing is the factor of realism i added the audio visual aspect to this simple text to text chat bot and created a virtual assistant .
This virtual assistant called DeepTalk is a Full pipline of Ai systems to bring life to this chat bot, first and for most is the chatbot system I used gpt3 from open ai to just make an illustration of the project the important is the visulization system , secondly we the output response text from this bot and pass it through a TTS system to form the wav form or the speech of the text the TTS system i used is based on transformers network i used this paper Neural Speech Synthesis with Transformer Network its advanced version of tactoron 2 which uses encoder/decoder transformer to generate mel spectrograms from a text and later will be transformed to wav forms useing Griffin-Lim Algorithm, last but not least is the MesTalk system developped by Facebook research get the generated wav file and .obj file describing the 3d avatra we want to animate the Meshtalk system have many properties its composed from 2 encoders one for audio feature extraction and the other for video animation feature extraction this 2 encoded passed though a multi perceptron network to form a categorical latent space this categorical aspect is important for time based data because its disentangle the the audio correlated features and uncorrelated features (e.g lips sync with audio and eyes movment) then the job of unet to input a mesh of the avatar and the latent space information and generate a sequence of meshes (i.g facial blend shapes),finally the rendering i used a framework Pytorch3d to genearte jpg pics for each generated mesh in the sequence and make a 30 dps video of the bot talking from a sequence of jpg images sync with the audio 

## 2) video to demonstrate the work 


https://user-images.githubusercontent.com/74627775/188270463-4ead2fe2-0a49-4a5a-a0a3-8d8ff542dc57.mp4

## 3) decencies and project setup for use

list of dependcies

* Pytorch
* pytorch 3d 
* numpy 
* librosa
* torchaudio
* scipy
* pickle
* tqdm
* trimesh
* unidecode
* ffmpeg-python
* iopath
* pip install -q espnet==0.10.6 pyopenjtalk==0.2 pypinyin==0.44.0 parallel_wavegan==0.5.4 gdown==4.4.0 espnet_model_zoo
* openai

## 4) weights for each model 

you can download weights from this link 

* put config.yaml and train.total_count.ave_10best.pth in for_espnet2 folder if you wanna use espnet2 as tts 
* put context_model.pkl,encoder.pkl and vertex_unet.pkl in for_mesh_animation
* put checkpoint_postnet_100000.pth.tar and checkpoint_transformer_160000.pth.tar if you want to use tts with transformer network 


## 5) how to use 

dependencies
```
import torch as th
import numpy as np
import torchaudio
from utils.helpers import *
from Models.context_model import *
from Models.encoders import *
from Models.vertex_unet import *
from utils.renderer import *


from utils.TTSMelUtils import spectrogram2wav
from utils.TTSTextUtils import text_to_sequence,string_chunks
from Models.TTSNetworks import ModelPostNet, Model


from utils.chatbotUtils import *

from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none

from scipy.io.wavfile import write
from collections import OrderedDict
from tqdm import tqdm
import pickle
import io
import librosa
import torch
import time 
from google.colab.patches import cv2_imshow
import cv2


max_len = 1000
SR = 22050
lang = 'English'
tag = 'kan-bayashi/ljspeech_vits'
vocoder_tag = "parallel_wavegan/ljspeech_parallel_wavegan.v1"
from animator.PredictMeshSequences import MeshSeq
from utils.chatbotUtils import show_video
from utils.chatbotUtils import *


```
init the system 
```

a  = MeshSeq("/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/MeshData/face_template.obj",
              "/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/MeshData/face_mean.npy",
              "/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/MeshData/face_std.npy",
              "/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/MeshData/forehead_mask.txt",
              "/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/MeshData/neck_mask.txt")


geom_unet,context_model,encoder,_,_,espnet2 = a.animator_model_setup()
```


loop for chat 
```
print("Loading Models is Done")
question = ""
print("\nStart the Chat with VRass (to quit type \"stop\")")
while True:
          print('setp1 : input your question:')
          question = input("Question: ")
          if question == "stop":
              break
          

          text = chat(question,start_chat_log).split("\n")[0]
          print("step1 is done")
          print('step2: start generating a speech from input text')
          text2voiceRes = a.get_espnet_prediction(espnet2,text)
          torchaudio.save(a.audio_path+"/testaudio1.wav",text2voiceRes[None,:].cpu(), sample_rate=16000)
          print('step2 is done')
          print('setp3 : generate a mesh seq or facial blendshapes sync with the generated audio')
          voice2AnimationRes = a.get_Animation_model_prediction(geom_unet,context_model,encoder,text2voiceRes[None,:])
          print('step3 is done')
          print('step4 : render the generated mesh seq to seq of jpg images')
          renderer = Renderer(a.template_verts_path)
          print('step4 is done')
          print('step 5 : transform the seq of jpg images to video given a frame rate')
          renderer.to_video(voice2AnimationRes,a.audio_path+"/testaudio1.wav",a.save_path+"/"+"test")
          print('setp5 is done')
          
```

in the end i provided this link to colab to show you code in more details [Use DeepTalk](https://colab.research.google.com/drive/1DZfBoaWp2Idf8Ym6wtwRklKde_N4J4wq#scrollTo=6GfnyluST__J) 
