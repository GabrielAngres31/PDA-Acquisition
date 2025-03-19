import os
import time
import typing as tp

import numpy as np
import torch

import src.data

import csv

def run_training_mbn(
  module: torch.nn.Module,  # MobileNet Classifier
  training_folders: src.data.Folders,
  epochs: int,
  learning_rate: float,
  batchsize: int,
  pos_weight: float,
  checkpointdir: str,
  table_out: str = None,
  validation_folders: src.data.Folders | None = None,
):
  checkpointdir = os.path.join(checkpointdir, time.strftime(f'%Y-%m-%d_%Hh-%Mm-%Ss'))
  os.makedirs(checkpointdir)

  loss_list = []

  module.train()
  optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
  dataset = src.data.Dataset_mbn(training_folders)
  loader = src.data.create_dataloader(dataset, batchsize, shuffle=True)
  vloader = None
  if validation_folders is not None:
    vdataset = src.data.Dataset(validation_folders)
    vloader = src.data.create_dataloader(vdataset, batchsize, shuffle=False)

  for e in range(epochs):
    loss = train_one_epoch_mbn(module, loader, optimizer, pos_weight)
    scheduler.step()
    if vloader is not None:
      vloss = validate_one_epoch_mbn(module, vloader)
    print(
      f'Epoch: {e:3d} | loss: {loss:.5f} {f"val.loss: {vloss:.5f}" if vloader else ""}'
    )
    torch.save(module, os.path.join(checkpointdir, f'last.e{e:03d}.pth'))
    if table_out:
      loss_list.append(f'Epoch: {e:3d} | loss: {loss:.5f} {f"val.loss: {vloss:.5f}" if vloader else ""},')

  if table_out:
    with open(f'{checkpointdir}{table_out}.csv', 'w', newline='') as csv_out:
      csv_out_writer = csv.writer(csv_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      [csv_out_writer.writerow(i) for i in loss_list]
  return module.eval()


def train_one_epoch_mbn(
  module: torch.nn.Module,
  loader: tp.Iterable,
  optimizer: torch.optim.Optimizer,
  pos_weight: float = 5.0,
) -> float:
  losses = []
  for i, [x, t] in enumerate(loader):
    optimizer.zero_grad()
    y = module(x)
    loss = torch.nn.functional.cross_entropy(
      y, t, pos_weight=torch.tensor(pos_weight)
    )
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    print(f'{i:3d}/{len(loader)} loss={loss.item():.4f}', end='\r')
  return np.mean(losses)


def validate_one_epoch_mbn(module: torch.nn.Module, loader: tp.Iterable) -> float:
    losses = []
    with torch.no_grad():
        for i, (x, t) in enumerate(loader):
            y = module(x)
            loss = torch.nn.functional.cross_entropy(y, t)
            losses.append(loss.item())
    return np.mean(losses)

