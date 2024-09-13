import argparse
import socket
import logging
import shutil
from typing import Tuple, Callable

import torch
import torch.nn as nn

from .util.parse import parse, setup_dir, prep_output
from .util.logger import init_logger, separator
from .generate_data.run_simulation import Simulation
from .generate_data.read_data import ReadData
from .model.flow import ConditionalInvertibleBlock
from .model.cnn import ConvNet3D
from .train.train import Training
from .eval.plotting import Plotting

def main():
    """
    Entry point of the program.
    Parses command line arguments and executes the corresponding function based on the subcommand.

    Usage:
        python __main__.py data <paramcard> [--verbose]
        python __main__.py train <paramcard> [--verbose]
        python __main__.py plot <paramcard> [--verbose]
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    
    data_parser = subparsers.add_parser("data")
    data_parser.add_argument("paramcard")
    data_parser.add_argument("--verbose", action="store_true")
    data_parser.set_defaults(func=data)
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("paramcard")
    train_parser.add_argument("--verbose", action="store_true")
    train_parser.set_defaults(func=train)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("paramcard")
    plot_parser.add_argument("--verbose", action="store_true")
    plot_parser.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)
    

def init(params)->Tuple[Callable, ConditionalInvertibleBlock, nn.Module, str]:
    """
    Function to initialize common classes
    """
    read_data = ReadData(height=140, width=140, img_length=2350).read
    flow = ConditionalInvertibleBlock(params).flow
    cnn = ConvNet3D(params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return read_data, flow, cnn, device

def data(args: argparse.Namespace) -> None:
    """
    Function to generate or convert the data to mock data
    """
    params = parse(args.paramcard)
    init_logger(fn=params['logging_name'], verbose=args.verbose)
    Simulation(params).main()

def train(args: argparse.Namespace) -> None:
    """
    Function to train the model
    """
    params = parse(args.paramcard)
    run_name = setup_dir(args.paramcard)
    params['name'] = run_name
    init_logger(fn=run_name, verbose=args.verbose)
    logging.info(f'{socket.gethostname()}: Node initialized, make sure it is where your data is!')
    read_data, flow, cnn, device = init(params)
    Training(params, flow, cnn, read_data, device).main()

def plot(args: argparse.Namespace) -> None:
    """
    Function to test the model and plot the output
    """
    params = parse(args.paramcard)
    run_name, plot_dir = prep_output(args.paramcard)
    params["name"] = run_name
    params['plot']['plot_dir'] = plot_dir
    init_logger(fn=plot_dir, verbose=args.verbose)
    read_data, flow, cnn, device = init(params)
    Plotting(params, flow, cnn, read_data, device).main()
 
if __name__ == "__main__":
    main()