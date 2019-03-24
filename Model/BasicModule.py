import torch
import time

class BasicModule(torch.nn.Module):
    """
    Adding save & load operation for the module

    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name=str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name
