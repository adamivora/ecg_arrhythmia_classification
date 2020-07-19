from copy import deepcopy
from time import time

import torch
from sklearn.metrics import f1_score
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from detection.preprocessing.transforms import Resample, Standardize, RandomFlip


def collate_fn(batch):
    """
    Custom batch collate function for variable-length 1D data. Zero-pads (appends zeros) to the length of the longest signal from
    the batch.

    :param batch: batch of variable-length signals
    :return: the padded batch
    """

    samples, labels, fs = zip(*batch)

    # the copy is needed because PyTorch crashes when trying to create a tensor from
    # an ndarray with a negative stride ¯\_(ツ)_/¯
    samples = [torch.from_numpy(x.copy()).unsqueeze(dim=1) for x in samples]

    samples = pad_sequence(samples, batch_first=True).transpose(1, 2)
    return samples, torch.tensor(labels, dtype=torch.long), torch.tensor(fs)


def get_transforms(training=True):
    """
    Get all the transforms for a dataset.

    :param training: True if the training transforms (data augmentation) should be included
    :return:
    """

    transform = transforms.Compose([
        Resample(fs=100),
        Standardize()
    ])
    if training:
        transform.transforms.append(RandomFlip(p=0.5))

    return transform


@torch.no_grad()
def predict(model, dataloader, device):
    """
    Predict the labels for a dataloader.

    :param model: the trained model
    :param dataloader: the dataloader to do inference on
    :param device: device to use for inference
    :return: all predictions for the `dataloader` samples
    """

    all_preds = torch.tensor([], dtype=torch.long, requires_grad=False)

    for batch in tqdm(dataloader):
        inputs = batch[0].float().to(device)

        outputs = model(inputs)
        all_preds = torch.cat((all_preds, outputs.argmax(dim=1).cpu()))

    return all_preds.numpy()


def train_model(model, opt, criterion, dataloaders, device, clip_gradient_norm=True, epochs=100):
    """
    Train a neural-network model for `epochs` epochs.

    :param model: a PyTorch `nn.Module` to train
    :param opt: the optimizer
    :param criterion: the loss criterion
    :param dataloaders: a dictionary of dataloaders, 'train' key is necessary, `val` recommended for validation
    :param device: the device to use for training, `cuda` recommended
    :param clip_gradient_norm: True if gradient clipping should be used. (On the difficulty of training Recurrent Neural Networks. https://arxiv.org/pdf/1211.5063.pdf)
    :param epochs: number of epochs to train
    :return: the best trained model by the validation f-score
    """

    phases = ['train', 'val']
    best_fscore = 0.0
    best_model = None

    for epoch in range(epochs):
        since = time()

        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            total_inputs = 0
            running_loss = 0.0
            all_preds = torch.tensor([], dtype=torch.long, requires_grad=False)
            all_labels = torch.tensor([], dtype=torch.long, requires_grad=False)

            for idx, batch in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                inputs, labels = batch[0].float().to(device), batch[1].to(device)

                opt.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    all_preds = torch.cat((all_preds, outputs.detach().argmax(dim=1).cpu()))
                    all_labels = torch.cat((all_labels, batch[1].detach()))
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        if clip_gradient_norm:
                            clip_grad_norm_(model.parameters(), 1)
                        opt.step()

                # print({f'batch': iii, f'{phase} batch loss': loss.item()})
                running_loss += loss.item() * inputs.size(0)
                total_inputs += inputs.size(0)

                if idx % 50 == 0 and phase == 'train':
                    print(f"{idx}: {running_loss / ((idx + 1) * inputs.size(0))}")
                    f_score = f1_score(all_labels.numpy(), all_preds.numpy(), average=None)
                    print(f"F-score: {f_score.mean()} - {f_score}")

            epoch_loss = running_loss / total_inputs
            epoch_fscore = f1_score(all_labels.numpy(), all_preds.numpy(), average=None)

            print(f'\n-----------------\nEpoch {epoch}\n')
            print(f'\t[{phase}] Loss: {epoch_loss:.4f}\n'
                  f'\tF-score: {epoch_fscore.mean():.4f} - {epoch_fscore}')
            print()
            print({'epoch': epoch, f'{phase} loss': epoch_loss, f'{phase} f-score': epoch_fscore.mean()})

            # deep copy the model
            if phase == 'val' and epoch_fscore.mean() > best_fscore:
                best_fscore = epoch_fscore.mean()
                best_model = deepcopy(model).cpu()
                print(f'new best fscore {epoch_fscore}')
        time_elapsed = time() - since
        print(f'Epoch {epoch} ended in {time_elapsed:.2f} seconds.')

    print(f'Best val fscore: {best_fscore:4f}')
    return best_model


def get_dataloader(dataset, batch_size, training, num_workers=8):
    """
    Get a dataloader for a dataset.

    :param dataset: dataset of the dataloader
    :param batch_size: batch size of the dataloader
    :param training: True if the dataset is used for training
    :return: torch.utils.DataLoader
    """

    weights = dataset.get_sample_weights()
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(weights, len(weights)) if training else None,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )


def get_dataloaders(dataset, batch_size):
    """
    Get dataloaders for `train` and `val` phases.

    :param dataset: the dataset to get dataloaders form
    :param batch_size: batch size of the dataloaders
    :return: a dictionary of dataloaders to use in the `train_model` function
    """
    return {phase: get_dataloader(dataset.get_split(phase), batch_size, phase == 'train') for phase in ['train', 'val']}
