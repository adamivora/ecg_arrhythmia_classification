from os import path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch import onnx
from torchvision import transforms

from .base import BaseModel
from .nn.common import get_dataloaders, get_dataloader, get_transforms, train_model, predict


class SignalBasedClassifier(BaseModel):
    """
    Wrapper class for all PyTorch neural network models based on the direct signal input.
    """
    cv = False

    def __init__(self, model_func, device, loss=nn.CrossEntropyLoss, optimizer=optim.Adam, batch_size=64,
                 epochs=100, weight_decay=0, optimizer_args=None, **model_args):
        super().__init__()
        if optimizer_args is None:
            optimizer_args = {}

        self.model_func = model_func
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.optimizer_args = optimizer_args
        self.model_args = model_args
        self.model = None

    def name(self):
        return self.model_func.__name__ + 'Classifier'

    def save(self, output_filename):
        torch.save(self, output_filename)

    def load(self, input_filename):
        return torch.load(input_filename)

    def score(self, dataset, cv=False):
        assert not cv, f'{type(self).__name__} does not support cross-validation scoring.'
        y_true = dataset.get_labels()
        y_pred = self.predict(dataset)

        score = f1_score(y_true, y_pred, average='macro')
        print(self.name(), ':', f1_score(y_true, y_pred, average=None))
        return score

    def predict(self, dataset):
        self.model.to(self.device)

        loader = get_dataloader(dataset, self.batch_size, training=False)
        y_pred = predict(self.model, loader, self.device)

        self.model.cpu()
        return y_pred

    def train(self, dataset, **train_args):
        self.model = self.model_func(num_classes=dataset.get_num_classes(), **(self.model_args))
        self.model.train().to(self.device)

        train_set, val_set, _ = dataset.get_splits(cv=False)
        train_set.transform = transforms.Compose(
            [train_set.transform,
             get_transforms(training=True)]) if train_set.transform is not None else get_transforms(training=True)
        val_set.transform = transforms.Compose(
            [val_set.transform,
             get_transforms(training=False)]) if val_set.transform is not None else get_transforms(training=False)

        loaders = get_dataloaders(dataset, batch_size=self.batch_size)
        self.model = train_model(self.model,
                    self.get_optimizer(),
                    self.loss(),
                    loaders,
                    self.device,
                    epochs=self.epochs,
                    **train_args)

    @torch.no_grad()
    def export(self, output_dir):
        assert self.model is not None, "Only a trained model can be exported."

        self.model.eval().cpu()

        dummy_input = torch.randn((1, 1, 1)).float()
        output_filename = path.join(output_dir, f'{self.name()}.onnx')
        onnx.export(self.model, dummy_input, output_filename, input_names=['input'], output_names=['output'],
                    do_constant_folding=True, dynamic_axes={'input': [0, 2], 'output': [0, 2]})

    def get_optimizer(self):
        """
        PyTorch default implementation of weight_decay in optimizer is erroneous,
        as weight decay should not be used for batchnorm layers.
        More in https://arxiv.org/pdf/1706.05350.pdf
        """
        param_dict = dict(self.model.named_parameters())
        bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
        rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]

        print("Weight decay NOT applied to BN parameters")
        opt = self.optimizer([
            {'params': bn_params, 'weight_decay': 0},
            {'params': rest_params, 'weight_decay': self.weight_decay}
        ], **(self.optimizer_args))
        return opt
