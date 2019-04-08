import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
from Dataset.data_utils import MyDataSet
from Model.BasicModules import AlexNet
from torchvision import transforms as T
from torch.autograd import Variable

class Solver:
    """
    A solver class including model, criterion & optimizer

    """
    def __init__(self, model, **kwargs):

        self.model = model
        self.criterion = kwargs.pop('criterion')
        self.optimizer = kwargs.pop('optimizer')
        self.batch_size = kwargs.pop('batch_size')
        self.max_epoch = kwargs.pop('max_epoch')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def val(self, val_dataloader):
        self.model.eval()
        num_correct = 0
        num_samples = 0
        for x, y in val_dataloader:
            x_var = Variable(x)

            scores = model(x_var)
            _, preds = scores.data.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        self.model.train()

    def train(self, train_data, val_data):

        train_dataloader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(dataset=val_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
        criterion = self.criterion
        optimizer = self.optimizer
        max_epoch = self.max_epoch

        for epoch in range(max_epoch):
            for i, (image, label) in enumerate(train_dataloader):
                # train model
                img = image.to(self.device)
                labels = label.to(self.device)

                optimizer.zero_grad()
                score = self.model(img)
                loss = criterion(score,labels)
                loss.backward()
                optimizer.step()

            self.model.save()

            self.val(val_dataloader)

if __name__=='__main__':
    transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])

    train_data = MyDataSet('./Dataset/', 'annotation.txt', transform)
    val_data = MyDataSet('./Dataset/', 'annotation.txt', transform, False, False)

    model = AlexNet()
    criterion = torch.nn.CrossEntropyLoss()
    optim = model.get_optimizer(lr=1e-4, weight_decay=1.0)
    solver = Solver(model=model, criterion=criterion, optimizer=optim, batch_size=30, max_epoch=10)

    solver.train(train_data, val_data)
