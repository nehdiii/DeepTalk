import openai as ai
from google.colab.patches import cv2_imshow
import cv2
from IPython.display import HTML
from base64 import b64encode

ai.api_key = "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"

completion = ai.Completion()

start_chat_log = """Human: Hello, I am Human.
    AI: Hello, human I am openai gpt3.
    Human: How are you?
    AI: I am fine, thanks for asking. 
    """


def chat(question,chat_log = None) -> str:
    if(chat_log == None):
        chat_log = start_chat_log
    prompt = f"{chat_log}Human: {question}\nAI:"
    response = completion.create(prompt = prompt, engine =  "davinci", temperature = 0.85,top_p=1, frequency_penalty=0, 
    presence_penalty=0.7, best_of=2,max_tokens=100,stop = "\nHuman: ")
    return response.choices[0].text

def modify_start_message(chat_log,question,answer) -> str:
    if chat_log == None:
        chat_log = start_chat_log
    chat_log += f"Human: {question}\nAI: {answer}\n"
    return chat_log

def show_video(video_path, video_width = 600):
   
  video_file = open(video_path, "r+b").read()
 
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video width={video_width} controls autoplay><source src="{video_url}"></video>""")
