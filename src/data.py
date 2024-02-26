import os
import shutil
import typing as tp

import numpy as np
import PIL.Image
import torch
import torchvision

FilePair  = tp.Tuple[str,str]
FilePairs = tp.List[FilePair]

def load_splitfile(splitfile:str) -> FilePairs:
    '''Read a .csv file containing paths to input images and annnotations.'''
    text  = open(splitfile, 'r').read()
    lines = text.split('\n')
    pairs = []
    for line in lines:
        paths = line.split(',')
        assert len(paths) == 2, f'Unexpected file format: {line}'
        
        inputpath, annotationpath = [p.strip() for p in paths]
        assert os.path.exists(inputpath), f'Could not find {inputpath}'
        assert os.path.exists(annotationpath), f'Could not find {annotationpath}'

        pairs.append((inputpath, annotationpath))
    return pairs


to_tensor = torchvision.transforms.ToTensor()

def load_image(path:str, mode:str) -> torch.Tensor:
    return to_tensor(PIL.Image.open(path).convert(mode))

def load_inputimage(path:str) -> torch.Tensor:
    return load_image(path, 'RGB')

def load_annotation(path:str) -> torch.Tensor:
    return load_image(path, 'L')

def save_image(path:str, imagedata:np.ndarray) -> None:
    assert imagedata.dtype == np.float32
    PIL.Image.fromarray((imagedata * 255).astype('uint8')).save(path)


def cache_file_pairs(
    pairs:       FilePairs, 
    destination: str, 
    patchsize:   int, 
    overlap:     int,
    clear:       bool = True,
) -> FilePairs:
    '''Slice input and annotation files into patches and cache them in a folder'''
    if clear:
        shutil.rmtree(destination, ignore_errors=True)
    os.makedirs(destination, exist_ok=True)
    open(os.path.join(destination, '.gitignore'), 'w').write('*')

    cached_pairs = []
    for inputpath, annotationpath in pairs:
        inputdata      = load_image(inputpath, 'L')
        annotationdata = load_annotation(annotationpath)
        assert inputdata.shape == annotationdata.shape

        inputpatches      = slice_into_patches_with_overlap(inputdata, patchsize, overlap)
        annotationpatches = slice_into_patches_with_overlap(annotationdata, patchsize, overlap)

        for i, [p_in, p_an] in enumerate(zip(inputpatches, annotationpatches)):
            in_destpath = os.path.join(
                destination, os.path.basename(inputpath)+f'.{i:04d}.input.tiff'
            )
            save_image(in_destpath, p_in[0].numpy())
            #PIL.Image.fromarray(p_in[0].numpy()).convert('RGB').save(in_destpath)

            an_destpath = os.path.join(
                destination, os.path.basename(inputpath)+f'.{i:04d}.annotation.tiff'
            )
            save_image(an_destpath, p_an[0].numpy())
            #PIL.Image.fromarray(p_an[0].numpy()).convert('L').save(an_destpath)

            cached_pairs.append( (in_destpath, an_destpath) )
    return cached_pairs


class Dataset:
    ''''Simple dataset class'''
    def __init__(self, filepairs:FilePairs):
        self.filepairs = filepairs
    
    def __len__(self):
        return len(self.filepairs)
    
    def __getitem__(self, i:int) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        inputfile, annotationsfile = self.filepairs[i]
        inputdata      = load_inputimage(inputfile)
        annotationdata = load_annotation(annotationsfile)
        assert inputdata.shape[-2:] == annotationdata.shape[-2:]

        return inputdata, annotationdata


def create_dataloader(
    ds:         torch.utils.data.Dataset, 
    batchsize:  int, 
    shuffle:    bool = False, 
    num_workers:int|tp.Literal['auto'] = 'auto', 
    **kw
) -> torch.utils.data.DataLoader:
    if num_workers == 'auto':
        num_workers = os.cpu_count() or 1
    return torch.utils.data.DataLoader(
        ds, 
        batchsize, 
        shuffle, 
        collate_fn      = getattr(ds, 'collate_fn', None),
        num_workers     = num_workers, 
        pin_memory      = True,
        worker_init_fn  = lambda x: np.random.seed(torch.randint(0,1000,(1,))[0].item()+x),
        **kw
    )



def grid_for_patches(
    imageshape:tp.Tuple[int,int]|torch.Size, patchsize:int, slack:int
) -> np.ndarray:
    '''Helper function for slicing images'''
    H,W       = imageshape[:2]
    stepsize  = patchsize - slack
    grid      = np.stack( 
        np.meshgrid( 
            np.minimum( np.arange(patchsize, H+stepsize, stepsize), H ), 
            np.minimum( np.arange(patchsize, W+stepsize, stepsize), W ),
            indexing='ij' 
        ), 
        axis=-1 
    )
    grid      = np.concatenate([grid-patchsize, grid], axis=-1)
    grid      = np.maximum(0, grid)
    return grid

def slice_into_patches_with_overlap(
    image:torch.Tensor, patchsize:int=1024, overlap:int=32
) -> tp.List[torch.Tensor]:
    '''Slice an image tensor into patches of size `patchsize` and overlap'''
    image     = torch.as_tensor(image)
    grid      = grid_for_patches(image.shape[-2:], patchsize, overlap)
    patches   = [image[...,i0:i1, j0:j1] for i0,j0,i1,j1 in grid.reshape(-1, 4)]
    return patches

def stitch_overlapping_patches(
    patches:    tp.List[torch.Tensor], 
    imageshape: tp.Tuple[int,int], 
    overlap:    int                     = 32, 
    out:        torch.Tensor|None       = None,
) -> torch.Tensor:
    '''Stitch patches previously sliced by `slice_into_patches_with_overlap()`'''
    patchsize = np.max(patches[0].shape[-2:])
    grid      = grid_for_patches(imageshape[-2:], patchsize, overlap)
    halfslack = overlap//2
    i0,i1     = (grid[grid.shape[0]-2,grid.shape[1]-2,(2,3)] - grid[-1,-1,(0,1)])//2
    d0 = np.stack( 
        np.meshgrid(
            [0]+[ halfslack]*(grid.shape[0]-2)+[i0]*(grid.shape[0]>1),
            [0]+[ halfslack]*(grid.shape[1]-2)+[i1]*(grid.shape[1]>1),
            indexing='ij' 
        ), 
        axis=-1
    )
    d1 = np.stack(
        np.meshgrid(     
            [-halfslack]*(grid.shape[0]-1)+[imageshape[-2]],      
            [-halfslack]*(grid.shape[1]-1)+[imageshape[-1]],
            indexing='ij'
        ), 
        axis=-1
    )
    d  = np.concatenate([d0,d1], axis=-1)
    if out is None:
        out = torch.empty(patches[0].shape[:-2] + imageshape[-2:], dtype=patches[0].dtype)
    for patch,gi,di in zip(patches, d.reshape(-1,4), (grid+d).reshape(-1,4)):
        out[...,di[0]:di[2], di[1]:di[3]] = patch[...,gi[0]:gi[2], gi[1]:gi[3]]
    return out

