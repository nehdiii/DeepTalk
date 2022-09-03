import torch
import trimesh
import requests
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures,
)
from animator.PredictMeshSequences import *
from utils.TTSMelUtils import spectrogram2wav
from scipy.io.wavfile import write
from utils.TTSTextUtils import text_to_sequence
import numpy as np
from Models.TTSNetworks import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import torch as th
import pickle
import io
import http

from trimesh.exchange.obj import export_obj

a  = MeshSeq("/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/MeshData/face_template.obj",
              "/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/MeshData/face_mean.npy",
              "/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/MeshData/face_std.npy",
              "/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/MeshData/forehead_mask.txt",
              "/content/gdrive/MyDrive/Colab_Notebooks/VirtualAssisstant/Data/MeshData/neck_mask.txt")


a.audio_vis_chat("test")


"""
for i in range(b.shape[0]):
    tensor_to_numpy = b[i].cpu().numpy()
    numpy_to_trimesh = trimesh.Trimesh(vertices=tensor_to_numpy)
    numpy_to_trimesh.export("C:/Users/nehdi/PycharmProjects/VirtualAssisstant/Data/meshs/face_obj_00{}.obj".format(i))
"""





"""
def load_checkpoint(step, model_name="transformer"):
    state_dict = th.load('C:/Users/nehdi/PycharmProjects/VirtualAssisstant/Data/pretrained_models/for_tts/checkpoint_%s_%d.pth.tar'% (model_name, step))
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict


def synthesis(text):
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpoint(160000,"transformer"))
    m_post.load_state_dict(load_checkpoint(100000,"postnet"))

    text = np.asarray(text_to_sequence(text))
    text = th.LongTensor(text).unsqueeze(0)
    text.cuda()

    mel_input = th.zeros([1,1,80]).cuda()
    pos_text = th.arange(1,text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.cuda()

    m = m.cuda()
    m_post = m_post.cuda()
    m.train(False)
    m_post.train(False)

    pbar = tqdm(range(max_len))

    with th.no_grad():
        for i in pbar:
            pos_mel = th.arange(1, mel_input.size(1) + 1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text.cuda(), mel_input.cuda(), pos_text.cuda(), pos_mel.cuda())
            mel_input = th.cat([mel_input, mel_pred[:, -1:, :]], dim=1)

        mag_pred = m_post.forward(postnet_pred)

    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    print(len(wav))

    write("C:/Users/nehdi/PycharmProjects/VirtualAssisstant/Data/audios/test4.wav", SR, wav)

"""




#synthesis("i dont know how to set up the max length of this model")*


