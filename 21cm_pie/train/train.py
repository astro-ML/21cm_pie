import logging
import glob
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import psutil
import torch
import torch.nn as nn
import torch.optim as optim

import FrEIA.framework as Ff

from ..util.logger import separator

class Training:
    """
    Class for training a network using the 21cm_pie package.

    Args:
        params (dict): A dictionary containing the training parameters.
        flow (Ff.SequenceINN): The flow model to be trained.
        cnn (nn.Module): The CNN model to be trained.
        data_reader (Callable): A function for reading the training data.
        device (str, optional): The device to be used for training. Defaults to 'cpu'.

    Attributes:
        flow (Ff.SequenceINN): The flow model to be trained.
        cnn (nn.Module): The CNN model to be trained.
        train_params (dict): The training parameters.
        batch_size (int): The batch size for training.
        n_dim (int): The dimensionality of the flow model.
        device (str): The device used for training.
        save_dir (str): The parent directory to save the trained models and loss data.
        data_reader (Callable): A function for reading the training data.

    Methods:
        setup_optimizer_and_scheduler: Sets up the optimizer and scheduler for training.
        find_data_paths: Finds the paths to the training and validation data.
        find_cnn_pred: Finds the predictions of the CNN model.
        train_epoch: Trains the model for one epoch.
        val_epoch: Evaluates the model on the validation data for one epoch.
        train_network: Trains the network.
        main: Main method for training the network.
    """

    def __init__(self, params: dict,
                 flow: Ff.SequenceINN,
                 cnn: nn.Module,
                 data_reader: Callable,
                 device: str = 'cpu'):
        self.flow = flow.to(device)
        self.cnn = cnn.to(device)
        self.train_params = params['train']
        self.batch_size = self.train_params['batch_size']
        self.n_dim = params['flow']['n_dim']
        self.device = device
        self.save_dir = params['name']+'/'
        self.data_reader = data_reader
        
        
    def setup_optimizer_and_scheduler(self) -> None:
        """
        Sets up the optimizer and scheduler for training.
        """
        lr = self.train_params['lr']
        scheduler_type = self.train_params.get('scheduler', 'StepLR')
        scheduler_params = self.train_params.get('scheduler_params', {})
        model = self.train_params['train_network']
        if model == 'flow':
            self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        elif model == 'cnn':
            self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=lr)
        elif model == 'both':
            self.optimizer = torch.optim.Adam([*self.flow.parameters(),*self.cnn.parameters()], lr=lr)
        else:
            raise ValueError(f"Unsupported model type: {model}")
        if scheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)
        elif scheduler_type == 'ExponentialLR':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, **scheduler_params)
        elif scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_params)
        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        logging.info(f"Using {scheduler_type} scheduler")
        
    def find_data_paths(self) -> Tuple[List[str], List[str]]:
        """
        Finds the paths to the training and validation data.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing the paths to the training and validation data.
        """
        data_path = self.train_params['data_path']
        data_paths = glob.glob(data_path + '/run*.npz')
        # sort the paths and take the first 4000
        data_paths = sorted(data_paths, key=lambda x: int(re.search(r'run(\d+)', x).group(1)))[:4000]
        trn_paths = data_paths[:int(0.9*len(data_paths))]
        val_paths = data_paths[int(0.9*len(data_paths)):]
        logging.info(f"Found {len(trn_paths)} training lightcones and {len(val_paths)} validation lightcones") 
        return trn_paths, val_paths
    
    def find_cnn_pred(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Finds the predictions of the Convolutional Neural Network (CNN) model.

        Returns:
            A tuple containing four numpy arrays:
            - cnn_pred_trn: Predictions of the CNN model on the training data.
            - cnn_label_trn: Labels of the training data.
            - cnn_pred_val: Predictions of the CNN model on the validation data.
            - cnn_label_val: Labels of the validation data.
        """
        try:
            # try loading the output of the CNN from the given path
            path = self.train_params['cnn_model_path']
            with np.load(path) as data:
                cnn_pred_trn = data['train_pred']
                cnn_label_trn = data['train_labels']
                cnn_pred_val = data['val_pred']
                cnn_label_val = data['val_labels']
            logging.info('Loaded CNN output')
        
        except:
            # Check if the CNN output is already saved in the directory
            try:
                path = self.save_dir + 'cnn_output.npz' 
                with np.load(path) as data:
                    cnn_pred_trn = data['train_pred']
                    cnn_label_trn = data['train_labels']
                    cnn_pred_val = data['val_pred']
                    cnn_label_val = data['val_labels']
                logging.info('Loaded CNN output')
            except:
                logging.info('No CNN output found. Running CNN to get predictions.')
                trn_paths, val_paths = self.find_data_paths()
                num_lightcones_trn = len(trn_paths)
                num_lightcones_val = len(val_paths)
                cnn_pred_trn = np.zeros((num_lightcones_trn, 6))
                cnn_label_trn = np.zeros((num_lightcones_trn, 6))
                cnn_pred_val = np.zeros((num_lightcones_val, 6))
                cnn_label_val = np.zeros((num_lightcones_val, 6))
                valid_indices_trn = []
                valid_indices_val = []
                # Some of the data may cause problems while loading
                # We need to check if the data is of the right shape
                for i, path in enumerate(trn_paths):
                    X, y = self.data_reader([path], self.device)
                    pred = self.cnn(X)
                    y_np = y.cpu().detach().numpy()
                    pred_np = pred.cpu().detach().numpy()
                    if y_np.shape == (1, 6) and pred_np.shape == (1, 6):
                        cnn_pred_trn[i] = pred_np
                        cnn_label_trn[i] = y_np
                        valid_indices_trn.append(i)
                    if i % 100 == 0:
                        logging.info(f'CNN progess 1: {i}/{num_lightcones_trn}')
                for i, path in enumerate(val_paths):
                    X, y = self.data_reader([path], self.device)
                    pred = self.cnn(X)
                    y_np = y.cpu().detach().numpy()
                    pred_np = pred.cpu().detach().numpy()
                    if y_np.shape == (1, 6) and pred_np.shape == (1, 6):
                        cnn_pred_val[i] = pred_np
                        cnn_label_val[i] = y_np
                        valid_indices_val.append(i)
                    if i % 100 == 0:
                        logging.info(f'CNN progress 2: {i}/{num_lightcones_val}')
                np.savez(self.save_dir+'cnn_output.npz', train_pred=cnn_pred_trn[valid_indices_trn],
                        train_labels=cnn_label_trn[valid_indices_trn], val_pred=cnn_pred_val[valid_indices_val],
                        val_labels=cnn_label_val[valid_indices_val])    
        separator()       
        return cnn_pred_trn, cnn_label_trn, cnn_pred_val, cnn_label_val
    
    def train_epoch(self, trn_paths: List[str],
                    model: str,
                    optimizer: optim.Optimizer,
                    loss_fn: Callable = nn.MSELoss(),
                    cnn_pred: np.ndarray = None,
                    cnn_label: np.ndarray = None
                  ) -> float:
        """
        Trains the model for one epoch.

        Args:
            trn_paths (List[str]): The paths to the training data.
            model (str): The type of model to train ('cnn', 'flow', or 'both').
            optimizer (optim.Optimizer): The optimizer to use for training.
            loss_fn (Callable, optional): The loss function to use for training. Defaults to nn.MSELoss().
            cnn_pred (np.ndarray, optional): The predictions of the CNN model. Defaults to None.
            cnn_label (np.ndarray, optional): The labels of the CNN model. Defaults to None.

        Returns:
            float: The average training loss per batch.
        """
        np.random.shuffle(trn_paths)
        num_lightcones = len(trn_paths)
        num_batches = num_lightcones/self.batch_size
        batch = 0
        iteration = 1
        trn_loss = 0.0
        while batch < num_lightcones:
            optimizer.zero_grad()
            # if the CNN is not actively trained, we don't need to run it again
            # simply take its output as the condition, which is the same every time
            if cnn_pred is None:
                X, y = self.data_reader(trn_paths[batch:batch+self.batch_size],self.device)
                pred = self.cnn(X)
            else:
                pred = torch.Tensor(cnn_pred[batch:batch+self.batch_size]).to(self.device)
                y = torch.Tensor(cnn_label[batch:batch+self.batch_size]).to(self.device)
            #check the RAM usage to not crash the cluster
            if psutil.virtual_memory()[2]>85:
                logging.info('Out of RAM. Abort training.')
                sys.exit()
            # Check which of the networks to train
            if model == 'cnn':
                loss = loss_fn(pred, y)
            else:
                z, jac = self.flow(y, c=[pred])
                loss = 0.5*torch.sum(z**2,1) - jac 
                loss = loss.mean() / self.n_dim
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            trn_loss += batch_loss
            if iteration % 100 == 0:
                logging.info( f"current batch loss: {batch_loss:>7f}  [{iteration:>5d}/{int(num_batches):>5d}]" )
                logging.info('RAM usage in percent: %3d', psutil.virtual_memory()[2])
            iteration += 1
            batch += self.batch_size
        trn_loss /= num_batches
        separator()
        logging.info( f"avg trn loss per batch: {trn_loss:>8f}" )
        return trn_loss
    
    def val_epoch(self, val_paths: List[str],
                  model: str,
                  loss_fn: Callable = nn.MSELoss(),
                  cnn_pred: np.ndarray = None,
                  cnn_label: np.ndarray = None
                  ) -> float:
        """
        Evaluates the model on the validation data for one epoch.

        Args:
            val_paths (List[str]): The paths to the validation data.
            model (str): The type of model to evaluate ('cnn', 'flow', or 'both').
            loss_fn (Callable, optional): The loss function to use for evaluation. Defaults to nn.MSELoss().
            cnn_pred (np.ndarray, optional): The predictions of the CNN model. Defaults to None.
            cnn_label (np.ndarray, optional): The labels of the CNN model. Defaults to None.

        Returns:
            float: The average validation loss per batch.
        """
        np.random.shuffle(val_paths)
        num_lightcones = len(val_paths)
        batch = 0
        num_batches = num_lightcones/self.batch_size
        val_loss = 0.0
        with torch.no_grad():
            while batch < num_lightcones:
                # if the CNN is not actively trained, we don't need to run it again
                # simply take its output as the condition, which is the same every time
                if cnn_pred is None:
                    X, y = self.data_reader(val_paths[batch:batch+self.batch_size],self.device)
                    pred = self.cnn(X)
                else:
                    pred = torch.Tensor(cnn_pred[batch:batch+self.batch_size]).to(self.device)
                    y = torch.Tensor(cnn_label[batch:batch+self.batch_size]).to(self.device)
                if model == 'cnn':
                    loss = loss_fn(pred, y)
                else:
                    z, jac = self.flow(y, c=[pred])
                    loss = 0.5*torch.sum(z**2,1) - jac
                    loss = loss.mean() / self.n_dim
                val_loss += loss.item()
                batch += self.batch_size
        val_loss /= num_batches
        logging.info( f"avg val loss per batch: {val_loss:>8f}" )
        return val_loss
    
    def plot_loss(self, trn_loss: List, val_loss: List, model: str) -> None:
        """
        Plots and saves the training and validation loss.

        Args:
            trn_loss (List): A list containing the training loss.
            val_loss (List): A list containing the validation loss.
            model (str): The type of model being trained ('cnn', 'flow', or 'both').
        """
        plt.plot(trn_loss, label='Training')
        plt.plot(val_loss, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.save_dir+'loss/loss_'+model+'.pdf')
        plt.close()
        np.save(self.save_dir+'loss/loss_'+model+'.npy', [trn_loss, val_loss])
        
    def save_models(self, model: str, epoch: int) -> None:
        """
        Saves the models.

        Args:
            model (str): The type of model to save ('cnn', 'flow', or 'both').
            epoch (int): The epoch number.
        """
        save_model_dir = self.save_dir + 'models/' + model + '/'
        if epoch == self.train_params['epochs']-1:
            logging.info('Saving final model')
            if (model == 'flow') or (model == 'both'):
                torch.save(self.flow.state_dict(), save_model_dir+f'flow_final.pth')
            if (model == 'cnn') or (model == 'both'):
                torch.save(self.cnn.state_dict(), save_model_dir+f'cnn_final.pth')
        else: 
            if (model == 'flow') or (model == 'both'):
                torch.save(self.flow.state_dict(), save_model_dir+f'flow_{epoch}.pth')
            if (model == 'cnn') or (model == 'both'):
                torch.save(self.cnn.state_dict(), save_model_dir+f'cnn_{epoch}.pth')
                
    def train_network(self) -> None:
        """
        Trains the network.
        """
        trn_paths, val_paths = self.find_data_paths()
        trn_loss = []
        val_loss = []
        model = self.train_params['train_network']
        save_model_dir = self.save_dir + 'models/' + model + '/'
        os.makedirs(save_model_dir, exist_ok=True)
        os.makedirs(self.save_dir + 'loss/', exist_ok=True)
        separator()
        logging.info(f"Training {model} network")
        separator()
        # if only the flow is trained, we need the CNN predictions
        if model == 'flow':
            cnn_pred_trn, cnn_label_trn, cnn_pred_val, cnn_label_val = self.find_cnn_pred()
        else:
            cnn_pred_trn = None
            cnn_label_trn = None
            cnn_pred_val = None
            cnn_label_val = None
        for epoch in range(self.train_params['epochs']):
            logging.info(f"Epoch {epoch+1}/{self.train_params['epochs']}")
            trn_loss.append(self.train_epoch(trn_paths, model, self.optimizer, cnn_pred=cnn_pred_trn, cnn_label=cnn_label_trn))
            val_loss.append(self.val_epoch(val_paths, model, cnn_pred=cnn_pred_val, cnn_label=cnn_label_val))
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss[-1])
            else:
                self.scheduler.step()
            self.plot_loss(trn_loss, val_loss, model)
            self.save_models(model, epoch)
            separator()
  
    def main(self) -> None:
        """
        Main method for training the network.
        """
        self.setup_optimizer_and_scheduler()
        self.train_network()
        logging.info('Training completed.')
        separator()
        
 
    
    
                
        
    
                    
            
        
        
        
        
        