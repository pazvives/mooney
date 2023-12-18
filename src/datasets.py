import glob
import torch
import torchvision
from PIL import Image

# NORM PARAMS PRE-CALCED WITH getNormalizationParams() over raw grayscale data
GRAYSCALE_MEAN = torch.tensor([0.4402])
GRAYSCALE_STD  = torch.tensor([0.2631])   
NORMALIZE      = torchvision.transforms.Normalize(GRAYSCALE_MEAN, GRAYSCALE_STD) 

# SET OF FINAL TRANSFORMS
RESIZE         = torchvision.transforms.Resize(size=(256, 256))
GRAYSCALE      = torchvision.transforms.Grayscale(num_output_channels=1)
GAUSSIAN_BLUR  = torchvision.transforms.GaussianBlur(kernel_size = (5,9), sigma = (0.1, 5.0))
BINARIZE       = torchvision.transforms.Lambda(lambda img: binarize(img, upper_threshold= .6, lower_threshold=.2))
TO_TENSOR      = torchvision.transforms.ToTensor()

def binarize(image: torch.Tensor, upper_threshold = .8, lower_threshold = .0) -> torch.Tensor:
    """ Binarizes an image according to an upper and lower threshold.
        If lower_threshold is not set (i.e. equals .0) then the results will be equivalent
        to having a single threshold. """
    
    binarized_image = image.clone().detach()
   
    # Set to 1 every pixel px lower than lower_threshold or higher than upper_threshold
    binarized_image[binarized_image > upper_threshold]  = 1.0
    binarized_image[binarized_image < lower_threshold]  = 1.0
   
    # Set to 0 everything else (ie between thresholds)
    binarized_image[binarized_image != 1.0]             = 0
    return binarized_image 


def getNormalizationParams(dataset):
  #stack all train images together into a tensor of shape (#images, #channels, height, width)
  from torch.utils.data import ConcatDataset
  x = torch.stack([sample[0] for sample in ConcatDataset([dataset])])

  #get the mean/std of each channel
  mean = torch.mean(x, dim=(0,2,3))
  std  = torch.std(x, dim=(0,2,3))
  return mean, std


class BinarizedVOCDataset(torch.utils.data.Dataset):

    def __init__(self, image_set = 'train', download = False, augmentations = None):
        self.to_grayscale  = torchvision.transforms.Compose(transforms = [ TO_TENSOR, RESIZE, GRAYSCALE])
        self.augmentations = augmentations
        self.bin_and_norm  = torchvision.transforms.Compose(transforms = [ BINARIZE ])
        self.dataset       = torchvision.datasets.VOCDetection( root      = '.',
                                                                year      = '2012',
                                                                image_set = image_set,
                                                                download  = download,
                                                                transform = self.to_grayscale)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ''' Note this returns tensors instead of PIL images '''
        grayscale_img   = self.dataset[idx][0] # Discarding the 'annotation' at [1] for now
        if self.augmentations:
            grayscale_img = self.augmentations(grayscale_img)
        binarized_img     = self.bin_and_norm(grayscale_img)
        return grayscale_img, binarized_img 



class HumanDataset(torch.utils.data.Dataset):

    def __init__(self, mooney_path = "humans/0_Mooney/", grayscale_path = "humans/1_Grayscale/"):
        self.mooney_path    = mooney_path
        self.grayscale_path = grayscale_path
        self.transforms          = torchvision.transforms.Compose(transforms = [ RESIZE ])
        mooney_file_list         = sorted(glob.glob(self.mooney_path + "*"))
        grayscale_file_list      = sorted(glob.glob(self.grayscale_path + "*"))
        self.images = []
        for mooney, grayscale in zip(mooney_file_list, grayscale_file_list):
            self.images.append([mooney, grayscale])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        mooney_path, grayscale_path = self.images[idx]
        mooney           = RESIZE(TO_TENSOR(Image.open(mooney_path)))
        grayscale        = RESIZE(TO_TENSOR(Image.open(grayscale_path)))
        return grayscale, mooney