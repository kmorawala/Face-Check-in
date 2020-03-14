import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('Test')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
workers = 0 if os.name == 'nt' else 4

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names))

# The following makes the app.
#importing open cv library for image/video processing
import cv2

# This will return video from the first webcam on the computer.
vidcap = cv2.VideoCapture(0)

#initializing variables
result, image = vidcap.read()
count = 0

#while there is a video result
while result is True:

    #create a jpg image with count number as the file name
    cv2.imwrite("frame%d.jpg" % count, image)
    # ret is a boolean regarding whether or not there was a return at all,
    # at the frame is each frame that is returned. If there is no frame, cap.read()
    # return None

    #update both variables
    result,image = vidcap.read()

    #Results printed and count updated, and the image is shown
    print('Read a new frame: ', result)
    count += 1
    cv2.imshow('image', image)

    # Runs once per frame and if the user presses key q, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()
