import os
import time
import typing as tp

import numpy as np
import torch
import torchvision
from torchvision import transforms as tf

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
    # model_ID:             str,
    table_out:            str,
    validation_filepairs: src.data.FilePairs|None = None,
):
    checkpointdir = os.path.join(checkpointdir, time.strftime(f'%Y-%m-%d_%Hh-%Mm-%Ss')) # f"{model_ID}_" + time.strftime(f'%Y-%m-%d_%Hh-%Mm-%Ss'))
    os.makedirs(checkpointdir)

    mean_loss_list = []
    all_loss_list = []
    mean_vloss_list = []
    all_vloss_list = []

    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    dataset   = src.data.Dataset(training_filepairs)
    loader    = src.data.create_dataloader(dataset, batchsize, shuffle=True)
    vloader   = None
    if validation_filepairs is not None:
        vdataset = src.data.Dataset(validation_filepairs)
        vloader  = src.data.create_dataloader(vdataset, batchsize, shuffle=False)
    print("Begun run_training")
    for e in range(epochs):
        loss_info = train_one_epoch(module, loader, optimizer, pos_weight)
        loss, loss_all = loss_info["mean"], loss_info["losses"]
        scheduler.step()
        if vloader is not None:
            vloss_info = validate_one_epoch(module, vloader)
            vloss, vloss_all = vloss_info["mean"], loss_info["losses"]
        
        print(
            f'Epoch: {e:3d} | loss: {loss:.5f}  { f"val.loss: {vloss:.5f}" if vloader else "" }'
        )
        torch.save(module, os.path.join(checkpointdir, f'last.e{e:03d}.pth'))
        if table_out:
            mean_loss_list.append(f'Epoch: {e:3d} | loss: {loss:.5f}  { f"val.loss: {vloss:.5f}" if vloader else "" },')
            [all_loss_list.append(f'{e}\t{l:.5f}') for l in loss_all]
            [all_vloss_list.append(f'{e}\t{l:.5f}') for l in vloss_all]
        print("write Z")


    
    if table_out:
        with open(f'{checkpointdir}{table_out}.csv', 'w', newline='') as csv_out:
            csv_out_writer = csv.writer(csv_out, delimiter=',',
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            [csv_out_writer.writerow([i]) for i in mean_loss_list]
        with open(f'{checkpointdir}{table_out}_all.csv', 'w', newline='') as csv_all_out:
            csv_out_all_writer = csv.writer(csv_all_out, delimiter=',',
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            [csv_out_all_writer.writerow([i]) for i in all_loss_list]
        with open(f'{checkpointdir}{table_out}_V.csv', 'w', newline='') as csv_out_v:
            csv_out_writer_v = csv.writer(csv_out_v, delimiter=',',
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            [csv_out_writer_v.writerow([i]) for i in mean_vloss_list]
        with open(f'{checkpointdir}{table_out}_all_V.csv', 'w', newline='') as csv_all_out_v:
            csv_out_all_writer_v = csv.writer(csv_all_out_v, delimiter=',',
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            [csv_out_all_writer_v.writerow([i]) for i in all_vloss_list]
        
    return module.eval()

import torchvision
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
        
        transform_code = np.random.randint(1,4)
        noise = transform_code >> 1
        blur = transform_code & 1

        # if np.random.random() < 0.5:
        if noise:
            gauss_noise = tf.v2.GaussianNoise()
            x_image=gauss_noise(x_image)
            pass #NOISING

        # if np.random.random() < 0.5:
        if blur:
            x_image = torchvision.transforms.functional.gaussian_blur(
                x_image, 
                kernel_size = int(np.random.choice([3, 5, 7])),
                sigma       = np.random.uniform(0.0, 3),
            )
        


    return new_x_batch, new_t_batch


def train_one_epoch(
    module:     torch.nn.Module, # TODO: ENSURE CORRECT MODULE (MBN)
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
    return {"mean":np.mean(losses), "losses":losses}

def validate_one_epoch(module:torch.nn.Module, loader:tp.Iterable) -> float:
    losses = []
    for i,[x,t] in enumerate(loader):
        with torch.no_grad():
            y    = module(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, t)
        losses.append(loss.item())
    return {"mean":np.mean(losses), "losses":losses}

# ----- MBN training code

def run_training_mbn(
    module:               torch.nn.Module, 
    training_folders:     torchvision.datasets.ImageFolder, 
    epochs:               int,
    learning_rate:        float,
    batchsize:            int,
    pos_weight:           float,
    checkpointdir:        str,
    # model_ID:             str,
    table_out:            str,
    validation_files, #: src.data.FilePairs|None = None,
):
    checkpointdir = os.path.join(checkpointdir, time.strftime(f'%Y-%m-%d_%Hh-%Mm-%Ss')) # f"{model_ID}_" + time.strftime(f'%Y-%m-%d_%Hh-%Mm-%Ss'))
    os.makedirs(checkpointdir)

    loss_list = []

    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    dataset   = torchvision.datasets.ImageFolder(training_folders, transform=src.data.to_tensor)
    loader    = src.data.create_dataloader_mbn(dataset, batchsize, shuffle=True)
    vloader   = None
    if validation_files is not None:
        vdataset = torchvision.datasets.ImageFolder(validation_files, transform=src.data.to_tensor)
        vloader  = src.data.create_dataloader_mbn(vdataset, batchsize, shuffle=False)

    for e in range(epochs):
        loss_info = train_one_epoch_mbn(module, loader, optimizer, pos_weight)

        loss, losses = loss_info["mean"], loss_info["losses"]

        scheduler.step()
        if vloader is not None:
            vloss = validate_one_epoch_mbn(module, vloader)
        print(
            f'Epoch: {e:3d} | loss: {loss:.5f}  { f"val.loss: {vloss:.5f}" if vloader else "" }'
        )
        torch.save(module, os.path.join(checkpointdir, f'last.e{e:03d}.pth'))
        if table_out:
            loss_list.append(f'Epoch: {e:3d} | loss: {loss:.5f}  { f"val.loss: {vloss:.5f}" if vloader else "" },')
            
            with open(f'{checkpointdir}{table_out}_cloud.csv', 'w', newline='') as csv_points_out:
                csv_points_writer = csv.writer(csv_points_out, delimiter=',',
                                    quotechar='\"', quoting=csv.QUOTE_MINIMAL)
                [csv_points_writer.writerow([f"Epoch: {e:3d}", l]) for l in losses]
    
    if table_out:
        with open(f'{checkpointdir}{table_out}.csv', 'w', newline='') as csv_out:
            csv_out_writer = csv.writer(csv_out, delimiter=',',
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            [csv_out_writer.writerow(i) for i in loss_list]
    return module.eval()

def augment_mbn(x_batch:torch.Tensor) -> torch.Tensor:
    '''Perform augmentations on inputs and annotation batches'''
    assert x_batch.ndim == 4

    new_x_batch = x_batch.clone()
    
    for i, (x_image) in enumerate(zip(x_batch)):
        # print(i)
        # print(x_image[0])
        k = np.random.randint(0,4)
        # x_image = tuple(torch.rot90(x_image[0], k, dims=(-1,-2)))
        x_image = torch.rot90(x_image[0], k, dims=(-1,-2))

        if np.random.random() < 0.5:
            x_image = torch.flip(x_image, dims=[-1])
        new_x_batch[i] = x_image
        
        # x_image = torchvision.transforms.functional.gaussian_blur(
        #     x_image, 
        #     kernel_size = 3,
        #     sigma       = np.random.uniform(0.0, 2.4),
        # )
    return new_x_batch


def train_one_epoch_mbn(
    module:     torch.nn.Module, 
    loader:     tp.Iterable, 
    optimizer:  torch.optim.Optimizer,
    pos_weight: float = 1.0,
) -> float:
    losses = []
    assert str(loader)[1:39] == "torch.utils.data.dataloader.DataLoader", f"{str(loader)[1:39]} is not 'torch.utils.data.dataloader.DataLoader'"
    # print("Well, is it subscriptable?")
    for i,[x,l] in enumerate(loader):
        # print("a")
        x  = augment_mbn(x)
        # print("a")
        optimizer.zero_grad()
        # print("a")
        y    = module(x)
        # print("a")
        loss = torch.nn.functional.cross_entropy(
            y, l
        )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f'{i:3d}/{len(loader)} loss={loss.item():.4f}', end='\r')
    return np.mean(losses)

def validate_one_epoch_mbn(module:torch.nn.Module, loader:tp.Iterable) -> float:
    losses = []
    for i,[x,l] in enumerate(loader):
        with torch.no_grad():
            y    = module(x)
        loss = torch.nn.functional.cross_entropy(y)
        losses.append(loss.item())
    return np.mean(losses)