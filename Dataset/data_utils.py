import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def generateDataFromVideo(path):
    video = cv2.VideoCapture(path)
    success, frame = video.read()
    cnt = 1
    file = open(file='annotation.txt', mode='w')

    while success:
        cv2.imwrite(filename='./data/{0}.jpg'.format(cnt), img=frame)
        cnt += 1
        success, frame = video.read()
        if cnt % 10 == 10:
            file.write('./data/{0}.jpg  1\n'.format(cnt))
        else:
            file.write('./data/{0}.jpg  0\n'.format(cnt))

class MyDataSet(Dataset):

    def __init__(self, root, datatxt, transform = None, target_transform = None):
        super(MyDataSet, self).__init__()
        file = open(root + datatxt, 'r')

        images = []
        for line in file:
            line = line.rstrip()
            words = line.split()
            images.append((words[0], int(words[1])))

        self.imgs = images
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, annot = self.imgs[index]
        image = cv2.imread(path)

        if self.transform is not None:
            image = self.transform(image)
        return image, annot

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    generateDataFromVideo('./data.mp4')
