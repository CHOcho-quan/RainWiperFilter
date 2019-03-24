import os
import cv2
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

def generateDataFromVideo(path):
    """
    Not so correct since the frequency of the rain wiper changes

    """
    video = cv2.VideoCapture(path)
    success, frame = video.read()
    cnt = 1
    wiperExist = 0
    file = open(file='annotation.txt', mode='w')

    while success:
        cv2.imwrite(filename='./data/{0}.jpg'.format(cnt), img=frame)
        cnt += 1
        success, frame = video.read()
        if (cnt - 4) % 37 == 0 or (wiperExist > 0):
            wiperExist = (wiperExist + 1) % 21
            file.write('./data/{0}.jpg  1\n'.format(cnt))
        else:
            file.write('./data/{0}.jpg  0\n'.format(cnt))

class MyDataSet(Dataset):

    def __init__(self, root, datatxt, transform = None, target_transform = None, train = True, test = False):
        super(MyDataSet, self).__init__()
        file = open(root + datatxt, 'r')

        images = []
        for line in file:
            line = line.rstrip()
            words = line.split()
            images.append((words[0], int(words[1])))

        imgs_num = len(images)
        self.test = test

        if self.test:
            self.imgs = images
        elif train:
            self.imgs = images[:int(0.7 * imgs_num)]
        else:
            self.imgs = images[int(0.7 * imgs_num):]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, annot = self.imgs[index]
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)
        return image, annot

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    # generateDataFromVideo('./data.mp4')

    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    data = MyDataSet('./', 'annotation.txt', transform)
    dataLoader = DataLoader(dataset=data, batch_size=3, shuffle=True)
    for batch_datas, batch_labels in dataLoader:
        print(batch_datas.size(),batch_labels.size())
    for img, label in data:
        print(img.size(), label)
