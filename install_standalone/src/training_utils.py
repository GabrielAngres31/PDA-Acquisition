import os
import time
import typing as tp

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as tfv2
# from torchvision import transforms as tf

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
            vloss, vloss_all = vloss_info["mean"], vloss_info["losses"]
        
        print(
            f'Epoch: {e:3d} | loss: {loss:.5f}  { f"val.loss: {vloss:.5f}" if vloader else "" }'
        )
        torch.save(module, os.path.join(checkpointdir, f'last.e{e:03d}.pth'))
        if table_out:
            mean_loss_list.append(f'Epoch: {e:3d} | loss: {loss:.5f}  { f"val.loss: {vloss:.5f}" if vloader else "" },')
            [all_loss_list.append(f'{e}\t{l:.5f}') for l in loss_all]
            if vloader:
                [all_vloss_list.append(f'{e}\t{v:.5f}') for v in vloss_all]


    
    if table_out:
        print(mean_loss_list)
        print(all_loss_list)
        print(mean_vloss_list if vloader else "NO VALIDATION")
        print(all_vloss_list if vloader else "NO VALIDATION")
        with open(f'{checkpointdir}{table_out}.csv', 'w', newline='') as csv_out:
            csv_out_writer = csv.writer(csv_out, delimiter=',',
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            [csv_out_writer.writerow([i]) for i in mean_loss_list]
        with open(f'{checkpointdir}{table_out}_all.csv', 'w', newline='') as csv_all_out:
            csv_out_all_writer = csv.writer(csv_all_out, delimiter=',',
                                quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            [csv_out_all_writer.writerow([i]) for i in all_loss_list]
        if vloader:
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

        # new_x_batch[i] = x_image
        # new_t_batch[i] = t_mask

        
        # transform_code = np.random.randint(0,2)
        # noise = transform_code >> 1
        # blur = transform_code & 1
        transform_code = np.random.randint(0,2)
        noise = transform_code >> 1
        blur = transform_code & 1

        # def add_black_squares(img_tensor, num_squares=3):

        #     height, width, channels = img_tensor.shape
        #     def add_single_square():
                
        #         x = torch.randint(0, width - 2, (1,)).item() 
        #         y = torch.randint(0, height - 2, (1,)).item()

        #         square = torch.zeros((3, 3, channels))
        #         img_tensor[y:y+3, x:x+3, :] = square

        #     for _ in range(num_squares):                
        #         add_single_square()
        #     return img_tensor
        
        # if np.random.random() < 0.05:
        #     x_image = add_black_squares(x_image, num_squares=3)

        # if np.random.random() < 0.5:

            # pass #NOISING

        # if np.random.random() < 0.5:
        if blur:
            x_image = torchvision.transforms.functional.gaussian_blur(
                x_image, 
                kernel_size = int(np.random.choice([3, 5, 7])),
                sigma       = np.random.uniform(0.0, 3),
            )
        
        new_x_batch[i] = x_image
        new_t_batch[i] = t_mask            
    
    return new_x_batch, new_t_batch


def train_one_epoch(
    module:     torch.nn.Module, # TODO: ENSURE CORRECT MODULE (MBN)
    loader:     tp.Iterable, 
    optimizer:  torch.optim.Optimizer,
    pos_weight: float = 5.0,
) -> float:

    import torch.nn.functional as F

    def weighted_binary_cross_entropy_with_logits(logits, targets, weights):
        """
        Calculates binary cross entropy with logits with weights.

        Args:
            logits: Predicted logits.
            targets: Ground truth targets (0 or 1).
            weights: Weights for each pixel.

        Returns:
            Weighted binary cross entropy loss.
        """
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_loss = loss * weights
        return torch.mean(weighted_loss)

    def distance_weights(image_size:tuple, center_weight:float=5.0, edge_weight:float=1.0):
        height, width = image_size
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        center_y, center_x = height // 2, width // 2
        
        distances = torch.sqrt( (y_coords - center_y) ** 2 + (x_coords - center_x) ** 2 )
        max_distance = torch.max(distances)
        normalized_distances = distances / max_distance
        
        weights = center_weight - (center_weight - edge_weight) * normalized_distances
        return weights
       
    losses = []
    for i,[x,t] in enumerate(loader):
        x,t  = augment(x,t)
        optimizer.zero_grad()
        y    = module(x)
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #     y, t, pos_weight=torch.tensor(pos_weight)
        # )
        loss = weighted_binary_cross_entropy_with_logits(
            y, t, distance_weights(tuple(x.size()[2:]))
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
        loss_fromfunc = train_one_epoch_mbn(module, loader, optimizer, pos_weight)
        loss_info = {"mean":np.mean(loss_fromfunc), "losses":loss_fromfunc}
        # print(loss_info)
        try:
            print("Attempting to allocate losses...")
            # print(loss_info)
            loss, losses = loss_info["mean"], loss_info["losses"]
        except IndexError:
            print(f"You've run into a strange error. Here's what we have:\n")
            try:    print(f"LOSS_INFO: {loss_info}")
            except: print("Couldn't get LOSS_INFO!")
            try:    print(f"[_mean]: {loss_info['mean']}")
            except: print("Couldn't get [_mean]!")
            try:    print(f"[_losses]: {loss_info['losses']}")
            except: print("Couldn't get [_losses]!")
        scheduler.step()
        if vloader is not None:
            vloss = validate_one_epoch_mbn(module, vloader)
        print(
            f'Epoch: {e:3d} | loss: {loss:.5f}  { f"val.loss: {vloss:.5f}" if vloader else "" }'
        )
        torch.save(module, os.path.join(checkpointdir, f'last.e{e:03d}.pth'))
        if table_out:
            loss_list.append([f'Epoch: {e:3d} | loss: {loss:.5f}  { f"val.loss: {vloss:.5f}" if vloader else "" }'])
            
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
        k = np.random.randint(0,4)
        x_image = torch.rot90(x_image[0], k, dims=(-1,-2))

        if np.random.random() < 0.5:
            x_image = torch.flip(x_image, dims=[-1])
        new_x_batch[i] = x_image
    return new_x_batch


def train_one_epoch_mbn(
    module:     torch.nn.Module, 
    loader:     tp.Iterable, 
    optimizer:  torch.optim.Optimizer,
    pos_weight: float = 1.0,
) -> float:
    losses = []
    assert str(loader)[1:39] == "torch.utils.data.dataloader.DataLoader", f"{str(loader)[1:39]} is not 'torch.utils.data.dataloader.DataLoader'"
    for i,[x,l] in enumerate(loader):
        x  = augment_mbn(x)
        optimizer.zero_grad()
        y    = module(x)
        loss = torch.nn.functional.cross_entropy(
            y, l
        )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f'{i:3d}/{len(loader)} loss={loss.item():.4f}', end='\r')
    return losses

def validate_one_epoch_mbn(module:torch.nn.Module, loader:tp.Iterable) -> float:
    losses = []
    for i,[x,l] in enumerate(loader):
        with torch.no_grad():
            y    = module(x)
        loss = torch.nn.functional.cross_entropy(y)
        losses.append(loss.item())
    return losses

# Fill In Guesser
def run_training_fill_in(
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
    loader    = src.data.create_dataloader_fill_in(dataset, batchsize, shuffle=True)
    vloader   = None
    if validation_files is not None:
        vdataset = torchvision.datasets.ImageFolder(validation_files, transform=src.data.to_tensor)
        vloader  = src.data.create_dataloader_fill_in(vdataset, batchsize, shuffle=False)

    for e in range(epochs):
        loss_info = train_one_epoch_fill_in(module, loader, optimizer, pos_weight)

        loss, losses = loss_info["mean"], loss_info["losses"]

        scheduler.step()
        if vloader is not None:
            vloss = validate_one_epoch_fill_in(module, vloader)
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

def augment_fill_in(x_batch:torch.Tensor) -> torch.Tensor:
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


def train_one_epoch_fill_in(
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
        x  = augment_fill_in(x)
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

def validate_one_epoch_fill_in(module:torch.nn.Module, loader:tp.Iterable) -> float:
    losses = []
    for i,[x,l] in enumerate(loader):
        with torch.no_grad():
            y    = module(x)
        loss = torch.nn.functional.cross_entropy(y)
        losses.append(loss.item())
    return np.mean(losses)