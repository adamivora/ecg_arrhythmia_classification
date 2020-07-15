from os import path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch import onnx
from torchvision import transforms

from .base import BaseModel
from .nn.common import get_dataloaders, get_dataloader, get_transforms, train_model, predict


class NNBasedClassifier(BaseModel):
    """
    Wrapper class for all PyTorch-based neural network models.
    """
    cv = False

    def __init__(self, model_generator, device, loss=nn.CrossEntropyLoss, optimizer=optim.Adam, batch_size=64,
                 epochs=100):
        super().__init__()
        self.model_generator = model_generator
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def name(self):
        return self.model_generator.__name__

    def score(self, dataset, cv=False):
        assert not cv, f'{type(self).__name__} does not support cross-validation scoring.'
        y_true = dataset.get_labels()
        y_pred = self.predict(dataset)

        score = f1_score(y_true, y_pred, average='macro')
        print(self.name(), ':', f1_score(y_true, y_pred, average=None))
        return score

    def predict(self, dataset):
        orig_transform = dataset.transform
        transform = get_transforms(training=False)
        dataset.transform = transforms.Compose([orig_transform, transform]) if orig_transform is not None else transform

        loader = get_dataloader(dataset, self.batch_size, training=False)
        y_pred = predict(self.model, loader, self.device)
        dataset.transform = orig_transform

        return y_pred

    def train(self, dataset, **model_args):
        self.model = self.model_generator(num_classes=dataset.get_num_classes(), **model_args)
        self.model.train().to(self.device)

        train_set, val_set, _ = dataset.get_splits(cv=False)
        train_set.transform = transforms.Compose(
            [train_set.transform,
             get_transforms(training=True)]) if train_set.transform is not None else get_transforms(training=True)
        val_set.transform = transforms.Compose(
            [val_set.transform,
             get_transforms(training=False)]) if val_set.transform is not None else get_transforms(training=False)

        loaders = get_dataloaders(dataset, batch_size=self.batch_size)
        train_model(self.model,
                    self.optimizer(self.model.parameters()),
                    self.loss(),
                    loaders,
                    self.device,
                    epochs=self.epochs)

    @torch.no_grad()
    def export(self, output_dir):
        assert self.model is not None, "Only a trained model can be exported."

        self.model.eval().cpu()

        dummy_input = torch.randn((1, 1, 1)).float()
        output_filename = path.join(output_dir, f'{self.name()}.onnx')
        onnx.export(self.model, dummy_input, output_filename, input_names=['input'], output_names=['output'],
                    do_constant_folding=True, dynamic_axes={'input': [0, 2], 'output': [0, 2]})
