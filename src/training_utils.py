import os
import time
import typing as tp

import numpy as np
import torch

import src.data

import csv

def run_training(
    module:               torch.nn.Module, 
    training_filepairs:   src.data.FilePairs,
    epochs:               int,
    learning_rate:        float,
    batchsize:            int,
    pos_weight:           float,
    checkpointdir:        str,
    table_out:            str,
    validation_filepairs: src.data.FilePairs|None = None,
):
    checkpointdir = os.path.join(checkpointdir, time.strftime(f'%Y-%m-%d_%Hh-%Mm-%Ss'))
    os.makedirs(checkpointdir)

    loss_list = []

    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    dataset   = src.data.Dataset(training_filepairs)
    loader    = src.data.create_dataloader(dataset, batchsize, shuffle=True)
    vloader   = None
    if validation_filepairs is not None:
        vdataset = src.data.Dataset(validation_filepairs)
        vloader  = src.data.create_dataloader(vdataset, batchsize, shuffle=False)
    
    for e in range(epochs):
        loss = train_one_epoch(module, loader, optimizer, pos_weight)
        scheduler.step()
        if vloader is not None:
            vloss = validate_one_epoch(module, vloader)
        print(
            f'Epoch: {e:3d} | loss: {loss:.5f}  { f"val.loss: {vloss:.5f}" if vloader else "" }'
        )
        torch.save(module, os.path.join(checkpointdir, f'last.e{e:03d}.pth'))
        if table_out:
            loss_list.append(f'Epoch: {e:3d} | loss: {loss:.5f}  { f"val.loss: {vloss:.5f}" if vloader else "" },')
    
    if table_out:
        with open(f'{checkpointdir}{table_out}.csv', 'w', newline='') as csv_out:
            csv_out_writer = csv.writer(csv_out, delimiter=',',
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            [csv_out_writer.writerow(i) for i in loss_list]
    return module.eval()


def augment(x_batch:torch.Tensor, t_batch:torch.Tensor) -> torch.Tensor:
    '''Perform augmentations on inputs and annotation batches'''
    assert x_batch.ndim == 4 and t_batch.ndim == 4 and len(x_batch) == len(t_batch)

    new_x_batch = x_batch.clone()
    new_t_batch = t_batch.clone()
    for i, (x_image, t_mask) in enumerate(zip(x_batch, t_batch)):
        k = np.random.randint(0,4)
        x_image = torch.rot90(x_image, k, dims=(-1,-2))
        t_mask  = torch.rot90(t_mask,  k, dims=(-1,-2))

        if np.random.random() < 0.5:
            x_image = torch.flip(x_image, dims=[-1])
            t_mask  = torch.flip(t_mask, dims=[-1])
        new_x_batch[i] = x_image
        new_t_batch[i] = t_mask
    return new_x_batch, new_t_batch


def train_one_epoch(
    module:     torch.nn.Module, 
    loader:     tp.Iterable, 
    optimizer:  torch.optim.Optimizer,
    pos_weight: float = 5.0,
) -> float:
    losses = []
    for i,[x,t] in enumerate(loader):
        x,t  = augment(x,t)
        optimizer.zero_grad()
        y    = module(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            y, t, pos_weight=torch.tensor(pos_weight)
        )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f'{i:3d}/{len(loader)} loss={loss.item():.4f}', end='\r')
    return np.mean(losses)

def validate_one_epoch(module:torch.nn.Module, loader:tp.Iterable) -> float:
    losses = []
    for i,[x,t] in enumerate(loader):
        with torch.no_grad():
            y    = module(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, t)
        losses.append(loss.item())
    return np.mean(losses)

