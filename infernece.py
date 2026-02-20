import cv2
import numpy as np
import time
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from facenet_pytorch import MTCNN

from dataset import DatasetLoader
from model import Net
import CONFIG

image_size = CONFIG.IMAGE_SIZE
use_gpu = torch.cuda.is_available()

# create the detector, using default weights

mtcnn = MTCNN(keep_all=True, device='cuda') if use_gpu else MTCNN(keep_all=True)

def inference_video(video_path, checkpoint_path, classes, webcam=False):
    model = Net(num_classes=CONFIG.NUM_CLASSES)

    if use_gpu:
        model = model.cuda()
    transform = transforms.Compose([transforms.Resize(image_size), transforms.PILToTensor()])
    
    assert os.path.exists(CONFIG.CHECKPOINT_PATH), "checkpoint not found"
    print("checkpoint to resume: ", checkpoint_path)
    tmp = torch.load(checkpoint_path)
    model.load_state_dict(tmp['state'])
    print("checkpoint restored !!!")

    cap = cv2.VideoCapture(0) if webcam else cv2.VideoCapture(video_path)

    while True:
        ret, frame_orig = cap.read()

        # convert the image to RGB
        # frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
        
        boxes, probs = mtcnn.detect(Image.fromarray(frame_orig))
        for i in range(len(boxes)):
            box = boxes[i]
            prob = probs[i]

            if prob <= 0.8:
                continue

            x1, y1, x2, y2 = [int(cor) for cor in box]
            face = frame_orig[y1:y2, x1:x2]

            face = transform(Image.fromarray(face))
            face = face / 255.

            if use_gpu:
                face = face.cuda()
            
            face = face.view(1, 3, image_size[0], image_size[1])
            with torch.no_grad():
                logps = model(face)
            
            # post process 
            ps = torch.exp(logps)
            probab = list(ps.cpu()[0])
            predicted_label = probab.index(max(probab))
            class_ = f"{classes[predicted_label].replace(classes[predicted_label][-1], '')}"

            frame_orig = cv2.rectangle(frame_orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame_orig = cv2.putText(frame_orig, class_, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("predictions", frame_orig)
        if cv2.waitKey(1) == 27:
            break


def inference_images(dataset_path, checkpoint_path, classes):
    model = Net(num_classes=classes)

    if use_gpu:
        model = model.cuda()

    assert os.path.exists(CONFIG.CHECKPOINT_PATH), "checkpoint not found"
    print("checkpoint to resume: ", checkpoint_path)
    tmp = torch.load(checkpoint_path)
    model.load_state_dict(tmp['state'])
    print("checkpoint restored !!!")

    transform = {
        "test": transforms.Compose([transforms.Resize(image_size), transforms.PILToTensor()])
    }

    test_dataset = DatasetLoader(dataset_path, transform, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
        shuffle=True
    )  

    fig, axes = plt.subplots(nrows=1, ncols=CONFIG.BATCH_SIZE, figsize=(4 * CONFIG.BATCH_SIZE, 4))
    
    dataiter = iter(test_loader)
    images, targets = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))

    for i in range(CONFIG.BATCH_SIZE):
        frame_orig, frame = images[i], images[i]
        gt = targets[i]

        if use_gpu:
            frame = frame.cuda()
        
        frame = frame.view(1, 3, image_size[0], image_size[1])
        with torch.no_grad():
            logps = model(frame)
        
        # post process 
        ps = torch.exp(logps)
        probab = list(ps.cpu()[0])
        predicted_label = probab.index(max(probab))
        title = f"GT: {gt}, predicted: {predicted_label}"

        frame_orig = frame_orig / 2 + 0.5     # unnormalize
        frame_orig = frame_orig.numpy()
        axes[i].imshow(np.transpose(frame_orig, (1, 2, 0)), interpolation='nearest', aspect='auto')
        axes[i].title.set_text(title)
    plt.show()


if __name__ == "__main__":
    inference_images(CONFIG.PATH_TO_DATASET_DIR, CONFIG.CHECKPOINT_PATH, CONFIG.NUM_CLASSES)
    
    # with open(CONFIG.PATH_TO_CLASSES_TXT, 'r') as f:
    #     classes = f.readlines()
    
    # inference_video(CONFIG.VIDEO_PATH, CONFIG.CHECKPOINT_PATH, classes, webcam=False)
