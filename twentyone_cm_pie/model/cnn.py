import logging
import sys

import torch
from torch import nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

class ConvNet3D(nn.Module):
    
    def __init__(self, params: dict,
                 in_ch: int = 1,
                 ch: int = 32,
                 N_parameter: int = 6, 
                 sigmoid: bool = False) -> None:
        """
        Initializes the CNN model.

        Args:
            params (dict): A dictionary containing the model parameters.
            in_ch (int, optional): Number of input channels. Defaults to 1.
            ch (int, optional): Number of output channels for the convolutional layers. Defaults to 32.
            N_parameter (int, optional): Number of output parameters. Defaults to 6.
            sigmoid (bool, optional): Whether to apply sigmoid activation to the output. Defaults to False.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, ch, kernel_size=(3,3,102), bias=True, stride=(1,1,102))
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=(3,3,2), bias=True)
        self.conv3 = nn.Conv3d(ch, 2*ch, kernel_size=(3,3,2), bias=True)
        self.conv3_zero = nn.Conv3d(2*ch, 2*ch, kernel_size=(3,3,2), bias=True, padding=(1,1,0))
        self.conv4 = nn.Conv3d(2*ch, 4*ch, kernel_size=(3,3,2), bias=True)
        self.conv4_zero = nn.Conv3d(4*ch, 4*ch, kernel_size=(3,3,2), bias=True, padding=(1,1,0))
        self.max = nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))
        self.avg = nn.AvgPool3d(kernel_size = (31,31,18))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, bias=True)
        self.linear2 = nn.Linear(128, 128, bias=True)
        self.linear3 = nn.Linear(128, 128, bias=True)
        self.sigmoid = sigmoid
        self.out = nn.Linear(128, N_parameter, bias=True)
        if params['cnn']['load']:
            if self.load_model(params['cnn']['model_location']):
                logging.info(f"Loaded cnn from {params['cnn']['model_location']}")
            else:
                logging.info(f"Failed to load cnn from {params['cnn']['model_location']}")
                sys.exit()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = self.max(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv3_zero(x))
        x = self.max(x)
        x = nn.ReLU()(self.conv4(x))
        x = nn.ReLU()(self.conv4_zero(x))
        x = self.avg(x)
        x = self.flatten(x)
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        x = nn.ReLU()(self.linear3(x))
        if self.sigmoid:
            x = nn.Sigmoid()(self.out(x)) 
        else:
            x = self.out(x)
        return x
    
    def load_model(self, location: str) -> bool:
        """
        Loads the model state from the specified location.

        Args:
            location (str): The file path to the saved model state.

        Returns:
            bool: True if the model was loaded successfully, False otherwise
        """
        try:
            self.load_state_dict(torch.load(location))
            return True
        except FileNotFoundError:
            return False
        
