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

## 4) weights for each model 

you can download weights from this link 

* put config.yaml and train.total_count.ave_10best.pth in for_espnet2 folder if you wanna use espnet2 as tts 
* put context_model.pkl,encoder.pkl and vertex_unet.pkl in for_mesh_animation
* put checkpoint_postnet_100000.pth.tar and checkpoint_transformer_160000.pth.tar if you want to use tts with transformer network 



