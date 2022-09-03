import torch as th
import numpy as np
import torchaudio
from utils.helpers import *
from Models.context_model import *
from Models.encoders import *
from Models.vertex_unet import *
from utils.renderer import *
# ------------------------------

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


class MeshSeq():
    def __init__(self,
                 template_verts_path,
                 mean_path,
                 stddev_path,
                 forhead_mask_path,
                 neck_mask_path,
                 pretarined_models_path="/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/pretrained_models/for_mesh_animation",
                 save_path="/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/videos"
                 ):
        self.template_verts_path = template_verts_path
        self.audio_path = "/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/audios"
        self.save_path = save_path
        self.template_verts = get_template_verts(template_verts_path)
        self.mean = th.from_numpy(np.load(mean_path))
        self.stddev = th.from_numpy(np.load(stddev_path))
        self.forhead_mask = th.from_numpy(load_mask(forhead_mask_path, dtype=np.float32)).cuda()
        self.neck_mask = th.from_numpy(load_mask(neck_mask_path, dtype=np.float32)).cuda()
        self.pretarined_models_path = pretarined_models_path

    def animator_model_setup(self):
        # models for animation
        geom_unet = VertexUnet(classes=128,
                               heads=16,
                               n_vertices=6172,
                               mean=self.mean,
                               stddev=self.stddev,
                               )
        geom_unet.load(self.pretarined_models_path)
        geom_unet.cuda().eval()
        context_model = ContextModel(classes=128,
                                     heads=16,
                                     audio_dim=128
                                     )
        context_model.load(self.pretarined_models_path)
        context_model.cuda().eval()
        encoder = MultimodalEncoder(classes=128,
                                    heads=16,
                                    expression_dim=128,
                                    audio_dim=128,
                                    n_vertices=6172,
                                    mean=self.mean,
                                    stddev=self.stddev,
                                    )
        encoder.load(self.pretarined_models_path)
        encoder.cuda().eval()

        # model for text 2 speech translation 

        tts_m = Model()
        tts_m_post = ModelPostNet()

        tts_m.load_state_dict(self.load_checkpoint(160000,"transformer"))

        tts_m_post.load_state_dict(self.load_checkpoint(100000,"postnet"))

        tts_m = tts_m.cuda()
        tts_m_post = tts_m_post.cuda()
        tts_m.train(False)
        tts_m_post.train(False)

        text2speechmodel = Text2Speech.from_pretrained(
              train_config="/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/pretrained_models/for_espnet2/config.yaml",
              model_file="/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/pretrained_models/for_espnet2/train.total_count.ave_10best.pth",
              device="cuda",
              # Only for Tacotron 2 & Transformer
              threshold=0.5,
              # Only for Tacotron 2
              minlenratio=0.0,
              maxlenratio=10.0,
              use_att_constraint=False,
              backward_window=1,
              forward_window=3,
              # Only for FastSpeech & FastSpeech2 & VITS
              speed_control_alpha=1.0,
              #  Only for VITS
              noise_scale=0.333,
              noise_scale_dur=0.333,
        )

        
        return geom_unet, context_model, encoder,tts_m,tts_m_post,text2speechmodel

    def load_checkpoint(self,step, model_name="transformer"):
        state_dict = th.load('/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/pretrained_models/for_tts/checkpoint_%s_%d.pth.tar'% (model_name, step))
        new_state_dict = OrderedDict()
        for k, value in state_dict['model'].items():
          key = k[7:]
          new_state_dict[key] = value

        return new_state_dict  

    def get_Animation_model_prediction(self,geom_unet,context_model,encoder,wav):
        audio = audio_chunking(wav, frame_rate=30, chunk_size=16000)
        with th.no_grad():
            audio_enc = encoder.audio_encoder(audio.cuda().unsqueeze(0))["code"]
            one_hot = context_model.sample(audio_enc, argmax=False)["one_hot"]
            T = one_hot.shape[1]
            geom = self.template_verts.cuda()[None, None, :, :].expand(-1, T, -1, -1).contiguous()
            result = geom_unet(geom, one_hot)["geom"].squeeze(0)
        # smooth results
        result = smooth_geom(result, self.forhead_mask)
        result = smooth_geom(result, self.neck_mask)
        return result

    def get_TTS_model_prediction(self,tts_m,tts_m_post,text):

      text = np.asarray(text_to_sequence(text))
      text = th.LongTensor(text).unsqueeze(0)
      text.cuda()

      mel_input = th.zeros([1,1,80]).cuda()
      pos_text = th.arange(1,text.size(1)+1).unsqueeze(0)
      pos_text = pos_text.cuda()

      with th.no_grad():
        for i in range(max_len):
            pos_mel = th.arange(1, mel_input.size(1) + 1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = tts_m.forward(text.cuda(), mel_input.cuda(), pos_text.cuda(), pos_mel.cuda())
            mel_input = th.cat([mel_input, mel_pred[:, -1:, :]], dim=1)

        mag_pred = tts_m_post.forward(postnet_pred)
      wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())

      return wav

    def get_espnet_prediction(self,espnetmodel,text):
      # synthesis
      with torch.no_grad():
          start = time.time()
          wav = espnetmodel(text)["wav"]

      return wav





    def animation_video_maker(self,file_name):
        geom_unet,context_model,encoder,tts_m,tts_m_post,_ = self.animator_model_setup()
        print("Load is Done")
        # get text fom user 
        print('give you inpute....')
        text = input() 
        text2voiceRes = self.get_TTS_model_prediction(tts_m,tts_m_post,text)
        text2voiceRes = librosa.resample(text2voiceRes, orig_sr=SR, target_sr=16000)
        print('text 2 voice done')
        torchaudio.save(self.audio_path+"/testaudio1.wav",th.from_numpy(text2voiceRes)[None,:], sample_rate=16000)
       
        print(th.from_numpy(text2voiceRes)[None,:].shape)
        voice2AnimationRes = self.get_Animation_model_prediction(geom_unet,context_model,encoder,th.from_numpy(text2voiceRes)[None,:])
        print('voice to mesh talk face video done')
        renderer = Renderer(self.template_verts_path)
        renderer.to_video(voice2AnimationRes,self.audio_path+"/testaudio1.wav",self.save_path+"/"+file_name)
        print("video saved")

    

    def animation_videos_maker_for_multiple_texts(self):
          geom_unet,context_model,encoder,tts_m,tts_m_post = self.animator_model_setup()
          print("Load is Done")
          # get text fom user 
          print('give you inpute....')
          text = input() 
          list_of_text_chunks = string_chunks(text,30)
          for i,text in enumerate(list_of_text_chunks):
            print(text)
            text2voiceRes = self.get_TTS_model_prediction(tts_m,tts_m_post,text)
            text2voiceRes = librosa.resample(text2voiceRes, orig_sr=SR, target_sr=16000)
            torchaudio.save(self.audio_path+"/testaudio1.wav",th.from_numpy(text2voiceRes)[None,:], sample_rate=16000)
            voice2AnimationRes = self.get_Animation_model_prediction(geom_unet,context_model,encoder,th.from_numpy(text2voiceRes)[None,:])
            renderer = Renderer(self.template_verts_path)
            renderer.to_video(voice2AnimationRes,self.audio_path+"/testaudio1.wav",self.save_path+"/"+"test_{}".format(i))
    
    def animation_video_maker_espnet2(self,file_name,geom_unet,context_model,encode,espnet2):
        geom_unet,context_model,encoder,_,_,espnet2 = self.animator_model_setup()
        print("Load is Done")
        # get text fom user 
        print('give you inpute....')
        text = input() 
        text2voiceRes = self.get_espnet_prediction(espnet2,text)
        #text2voiceRes = librosa.resample(text2voiceRes, orig_sr=SR, target_sr=16000)
        print('text 2 voice done')
        torchaudio.save(self.audio_path+"/testaudio1.wav",text2voiceRes[None,:].cpu(), sample_rate=16000)
        

        voice2AnimationRes = self.get_Animation_model_prediction(geom_unet,context_model,encoder,text2voiceRes[None,:])
        print('voice to mesh talk face video done')
        renderer = Renderer(self.template_verts_path)
        renderer.to_video(voice2AnimationRes,self.audio_path+"/testaudio1.wav",self.save_path+"/"+file_name)
        print("video saved")

    def audio_vis_chat(self,file_name):
      geom_unet,context_model,encoder,_,_,espnet2 = self.animator_model_setup()
      print("Loading Models is Done")
      question = ""
      print("\nStart the Chat with VRass (to quit type \"stop\")")
      while True:
          question = input("Question: ")
          if question == "stop":
              break
          text = chat(question,start_chat_log).split("\n")[0]
          text2voiceRes = self.get_espnet_prediction(espnet2,text)
          torchaudio.save(self.audio_path+"/testaudio1.wav",text2voiceRes[None,:].cpu(), sample_rate=16000)
          voice2AnimationRes = self.get_Animation_model_prediction(geom_unet,context_model,encoder,text2voiceRes[None,:])
          renderer = Renderer(self.template_verts_path)
          renderer.to_video(voice2AnimationRes,self.audio_path+"/testaudio1.wav",self.save_path+"/"+file_name)
          
          
          





    
          

    




