from flask import Flask, request
from flask_restful import  Resource
import numpy as np
import torch
import base64, ast
import cv2
import numpy as np
from utils.utils_crop import *
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from utils.net import Net
import torch.nn.functional as F

from PIL import Image



app = Flask(__name__)


# Set up environment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_crop = DetectMultiBackend("yolov3.pt", device=device, dnn=False)

model_detect = Net(num_class=1055)
model_detect.load_state_dict(torch.load("weights.pt", map_location=torch.device('cpu'))["state_dict"])
model_detect.eval()

@app.route('/', methods=["POST"])
def post():


    # use parser and find the user's query
    image2 = request.get_data()
    image2 = image2.decode('utf-8')

    mydata = ast.literal_eval(image2)

    base = mydata['image']

    imgdata = base64.b64decode(base)
    filename = 'whaleee.jpg' 
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()
    nparr = np.fromstring(imgdata, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Croping fluke 
    try:
        croped_img = crop_fluke(model_crop, device, img)
    except:
        croped_img = img    

    # Detecting whale
    labels0 = pd.read_csv("example_dataset.csv")
    labels0 = labels0.groupby("Id_int")["Id"].first().reset_index()
    labels0 = labels0.Id
    labels0 = np.array(labels0)

    image = np.array(croped_img)
    image = image.astype('uint8')

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    image = data_transform(image)
    image = image.expand(1,3,224,224)

    with torch.no_grad():
            logits, _, _ = model_detect.forward(x=image, is_infer=True)

            for logit in logits:
                prob = F.sigmoid(logit)
                prob = np.asarray(prob)
                top = np.argmax(prob)
                label = labels0[top]
        
    return label


# Setup the Api resource routing here
# Route the URL to the resource



if __name__ == '__main__':
    app.run(debug=False, port=5000)

