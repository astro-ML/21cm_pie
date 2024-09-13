import numpy as np
import logging
from typing import Tuple, List
import torch

class ReadData():
    """
    A class for reading and preparing data for training.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        img_length (int): The length of the image.

    Attributes:
        height (int): The height of the image.
        width (int): The width of the image.
        img_length (int): The length of the image.
        paras (int): The number of parameters.

    Methods:
        prepare(files: List[str]) -> Tuple[np.ndarray, np.array]:
            Prepares the data by reading and normalizing it.

        read(paths: List[str], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
            Reads the data and converts it to PyTorch tensors.

    """

    def __init__(self, height: int, width: int, img_length: int):
        self.height = height
        self.width = width
        self.img_length = img_length
        self.paras = 6

    def prepare(self, files: List[str], hdf=False) -> Tuple[np.ndarray, np.array]:
        """
        Prepares the data by reading and normalizing it.

        Args:
            files (List[str]): A list of file paths.

        Returns:
            Tuple[np.ndarray, np.array]: A tuple containing the image data and label data.

        """
        n_cones = len(files)
        image = np.zeros((n_cones, self.height, self.width, self.img_length))
        label = np.zeros((n_cones, self.paras))
        valid_indices = []
        # Some files may be corrupted, so we need to skip them
        for i in range(len(files)):
            try:
                cones = np.load(files[i])
                image[i] = cones['image']
                label[i] = cones['label']
            except:
                logging.info('bad file, zip:', files[i]) 
        # normalize the data
        mean = 0
        norm = 1250
        image = (image - mean) / norm
        labelMean = np.array([0.3, 0.2, 100, 38, 4, 10])
        labelNorm = np.array([9.7, 0.2, 1400, 4, 1.3, 240])
        for j in range(self.paras):
            label[:, j] = (label[:, j] - labelMean[j]) / labelNorm[j]
        return image[valid_indices], label[valid_indices], end-start

    def read(self, paths: List[str], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reads the data and converts it to PyTorch tensors.

        Args:
            paths (List[str]): A list of file paths.
            device (str): The device to move the tensors to.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor and label tensor.

        """
        # read in all the files matching the given pattern
        image, label = self.prepare(paths)
        image_tensor = torch.Tensor(image.astype('float32'))
        label_tensor = torch.Tensor(label.astype('float32'))
        # add a channel dimension to the image tensor
        image_tensor = image_tensor.unsqueeze(1)
        return image_tensor.to(device), label_tensor.to(device)
        
            
            
            
            
            
           
