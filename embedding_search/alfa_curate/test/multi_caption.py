import decord
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel
import subprocess
import io

# load model and pre-processor
model = AutoModel.from_pretrained("nvidia/Cosmos-Embed1-448p", trust_remote_code=True).to("cuda", dtype=torch.bfloat16)
preprocess = AutoProcessor.from_pretrained("nvidia/Cosmos-Embed1-448p", trust_remote_code=True)

# load mock data
video_url = "https://upload.wikimedia.org/wikipedia/commons/3/3d/Branko_Paukovic%2C_javelin_throw.webm"
subprocess.check_call(["wget", "-O", "/tmp/javelin_throw.mp4", video_url])
reader = decord.VideoReader("/tmp/javelin_throw.mp4")
frame_ids = np.linspace(0, len(reader)-1, 8, dtype=int).tolist()
frames = reader.get_batch(frame_ids).asnumpy()
batch = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))  # BTCHW
captions = [
    "a person riding a motorcycle in the night",
    "a car overtaking a white truck",
    "a video of a knight fighting with a sword",
    "a man wearing red spandex throwing a javelin",
    "a young man javelin throwing during the evening", # distractor
    "a man throwing a javelin with both hands", # distractor
]

# video and text processing
video_inputs = preprocess(videos=batch).to("cuda", dtype=torch.bfloat16)
video_out = model.get_video_embeddings(**video_inputs)
text_inputs = preprocess(text=captions).to("cuda", dtype=torch.bfloat16)
text_out = model.get_text_embeddings(**text_inputs)

# ranking and argmax
probs = (torch.softmax(model.logit_scale.exp() * video_out.visual_proj @ text_out.text_proj.T, dim=-1))[0]
print(captions[probs.argmax()])
