import glob
import os
import logging
import re
from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import binom

import torch
import torch.nn as nn
import getdist
from getdist import plots, MCSamples
import FrEIA.framework as Ff

from ..util.logger import separator

class Plotting():
    """
    A class for plotting and analyzing data.

    Args:
        params (dict): A dictionary containing parameters for plotting.
        flow (Ff.SequenceINN): A sequence of invertible neural networks.
        cnn (nn.Module): A neural network module.
        data_reader (Callable): A callable object for reading the lightcone data.
        device (str, optional): The device to use for computation. Defaults to 'cpu'.

    Attributes:
        parameters (list): A list of cosmological parameter information.
        params (dict): A dictionary containing parameters for plotting.
        data_reader (Callable): A callable object for reading the lightcone data.
        flow (Ff.SequenceINN): A sequence of invertible neural networks.
        cnn (nn.Module): A neural network module.
        device (str): The device used for computation.
        test_paths (list): A list of test data paths.
        output_dir (str): The output directory for plots.
        num_lightcones (int): The number of lightcones.
        cnn_pred (np.ndarray): The predicted output of the CNN.
        label (np.ndarray): The true labels.

    Methods:
        find_cnn_output: Tries to load the output of the CNN or runs it once.
        rescale: Rescales the parameters as they were normalized for training.
        name_to_index: Converts a name or index to the corresponding index or name.
        credible_interval: Returns the alpha-credible interval limits for a distribution.
        calc_statistics: Calculates statistics for the data.
        plot_calibration: Makes calibration plots.
        plot_rank_statistic: Makes rank statistic plots.
        sample_fiducial: Samples fiducial data.
        plot_corner: Plots a corner plot.

    """

    def __init__(self, 
                 params: dict,
                 flow: Ff.SequenceINN,
                 cnn: nn.Module,
                 data_reader: Callable,
                 device: str = 'cpu'
                 ) -> None:
        """
        Initializes the Plotting class.

        Args:
            params (dict): A dictionary containing parameters for plotting.
            flow (Ff.SequenceINN): A sequence of invertible neural networks.
            cnn (nn.Module): A neural network module.
            data_reader (Callable): A callable object for reading data.
            device (str, optional): The device to use for computation. Defaults to 'cpu'.

        """
        self.parameters=[["WDM",0.3,10,r"$m_{WDM}$",r"m_{WDM}"],["OMm",0.2,0.4,r"$\Omega_m$",r"\Omega_m"],
                    ["E0",100,1500,r"$E_0$",r"E_0"],["LX",38,42,r"$L_X$",r"L_X"],
                    ["Tvir",4,5.3,r"$T_{vir}$",r"T_{vir}"],["Zeta",10,250,r"$\zeta$",r"\zeta"]]
        self.params = params['plot']
        self.data_reader = data_reader
        self.flow = flow.to(device)
        self.cnn = cnn.to(device)
        self.device = device 
        test_paths = glob.glob(self.params['data_path']+'/run*.npz')
        self.test_paths = sorted(test_paths, key=lambda x: int(re.search(r'run(\d+)', x).group(1)))[4000:]
        logging.info(f"Found {len(self.test_paths)} test lightcones")
        self.output_dir = self.params['plot_dir'] + '/'
        self.num_lightcones = len(self.test_paths)  
              
    def find_cnn_output(self, save_name:str = 'cnn_test_output.npz') -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the output of the CNN model on the test data or runs it once.

        Args:
            save_name (str, optional): Name of the file to save the CNN output. Defaults to 'cnn_test_output.npz'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the CNN predictions and the corresponding labels.
        """
        try:
            with np.load(self.params['cnn_pred_path']) as data:
                cnn_pred = data['cnn_pred']
                label = data['label']
                separator()
                logging.info("Loaded CNN output")
                separator()
        except:
            separator()
            logging.info("Running CNN on test data")
            cnn_pred = np.zeros((self.num_lightcones, 6))
            label = np.zeros((self.num_lightcones, 6))
            for i, path in enumerate(self.test_paths):
                X, y = self.data_reader([path], self.device)
                pred = self.cnn(X)
                cnn_pred[i] = pred.detach().cpu().numpy()
                label[i] = y.detach().cpu().numpy()
                if i % 100 == 0:
                    logging.info(f"Processed {i} lightcones")
            np.savez(self.output_dir+save_name, cnn_pred=cnn_pred, label=label)
            logging.info("Saved and loaded CNN output")
            separator()
        return cnn_pred, label

    def rescale(self, y: np.array) -> np.array:
        """
        Rescales the parameters as they were normalized for training.

        Parameters:
        - y (np.array): The array of parameters to be rescaled.

        Returns:
        - np.array: The rescaled array of parameters.
        """
        y_rescale = np.zeros(y.shape)
        for para in range(len(self.parameters)):
            y_rescale[:,para] = [x*(self.parameters[para][2]-self.parameters[para][1])+self.parameters[para][1] for x in y[:,para]]
        return y_rescale
    
    def name_to_index(self, direction: str = 'forward',
                      name: int = 4692,
                      index: int = 0) -> int:
        """
        Converts a name or index to the corresponding index or name, depending on the direction.
        The name corresponds to the number in the file name, e.g. 'run4692.npz'.
        The index corresponds to the index in the test_paths list.

        Args:
            direction (str): The direction of conversion. Can be either 'forward' or 'backward'.
            name (int): The name to convert. Default is 4692.
            index (int): The index to convert. Default is 0.

        Returns:
            int: The converted index or name.

        Raises:
            None

        """
        if direction == 'forward':
            run_name = 'run' + str(name) + '.npz'
            try:
                index = self.test_paths.index(self.params['data_path'] + '/' +  run_name)
            except:
                logging.error(f"Lightcone {run_name} not found, return random index")
                index = np.random.randint(self.num_lightcones)
            return index
        elif direction == 'backward':
            run_name = os.path.basename(self.test_paths[index])
            name = int(run_name[3:-4])
            return name
    
    def credible_interval(self, x: np.ndarray ,alpha: float, index: int) -> Tuple[float, float]:
        """
        Calculate the credible interval for a given dataset.

        Parameters:
        x (np.ndarray): The samples for which the credible interval is calculated.
        alpha (float): The desired confidence level (between 0 and 1).
        index (int): The index of the dataset, corresponding to a parameter.

        Returns:
        Tuple[float, float]: The lower and upper bounds of the credible interval.
        """
        x_mc = MCSamples(samples=x)
        grid = x_mc.get1DDensityGridData(index)
        low, up = grid.getLimits([alpha])[0:2] 
        return low, up
    
    def calc_statistics(self, 
                        output_name: str,
                        sample_size: int = 10000,
                        cal_error: bool = True,
                        rank_stat: bool = True,
                        mean_metrics: bool = True,
                        ) -> None:
        """
        Calculate test statistics for the given test data and trained networks.

        Args:
            output_name (str): The name of the output file.
            sample_size (int, optional): The number of samples to generate. Defaults to 10000.
            cal_error (bool, optional): Whether to calculate calibration error. Defaults to True.
            rank_stat (bool, optional): Whether to calculate rank statistics. Defaults to True.
            mean_metrics (bool, optional): Whether to calculate mean metrics (r2 and nrmse). Defaults to True.

        Returns:
            None
        """
        self.cnn_pred, self.label = self.find_cnn_output() 
        logging.info('Calculating statistics')
        names = [self.parameters[para][3] for para in range(len(self.parameters))]
        lab = [self.parameters[para][4] for para in range(len(self.parameters))]
        mean = np.zeros(self.cnn_pred.shape)
        lower = np.zeros(self.cnn_pred.shape)
        upper = np.zeros(self.cnn_pred.shape)
        rank = None
        cal_err = None
        r2_mean = None
        nrmse = None
        if rank_stat:
            rank = np.zeros(self.cnn_pred.shape)
        if cal_error:
            bin_alpha = 100
            alpha = np.linspace(0.01,0.99,bin_alpha)
            alpha_0 = np.zeros((self.cnn_pred.shape[0],bin_alpha,len(self.parameters)))
        
        for i in range(self.cnn_pred.shape[0]):
            z = torch.randn((sample_size, len(self.parameters))).to(self.device)
            samples, _ = self.flow(z, c = [torch.Tensor(self.cnn_pred[i]).repeat((sample_size,1)).to(self.device)], rev=True)
            samples = samples.detach().cpu().numpy()
            if rank_stat:
                for j in range(len(self.parameters)):
                    rank[i,j] = (samples[:,j]<self.label[i,j]).sum()
            if cal_error:
                j = 0
                for a in alpha:
                    for para in range(len(self.parameters)):
                        low, up = self.credible_interval(samples,a,para) 
                        if (low<self.label[i,para]) & (self.label[i,para]<up):
                            alpha_0[i,j,para] = 1
                    j += 1      
            samp_mc = MCSamples(samples=self.rescale(samples), names=names, labels=lab, settings={'smooth_scale_2D':0.5,'smooth_scale_1D':0.5})
            stats = samp_mc.getMargeStats()
            for para,_ in enumerate(self.parameters):
                mean[i,para] = stats.parWithName(names[para]).mean
                lower[i,para] = stats.parWithName(names[para]).limits[0].lower
                upper[i,para] = stats.parWithName(names[para]).limits[0].upper
                
            if i % 100 == 0:
                logging.info(f"Processed {i} lightcones")
                
        if cal_error:
            alpha_0_bar=np.mean(alpha_0,axis=0)
            cal_err = np.mean(np.absolute(alpha_0_bar.T-alpha),axis=1)
        
        label_rescale = self.rescale(self.label)
        if mean_metrics:
            r2_mean = np.zeros(len(self.parameters))
            nrmse = np.zeros(len(self.parameters))
            for para in range(len(self.parameters)):
                # Calculate the R2-score for each parameter
                average = sum(label_rescale[:, para]) / len(label_rescale[:, para])
                dividend = sum((x - y) ** 2 for x, y in zip(label_rescale[:, para], mean[:, para]))
                divisor = sum((x - average) ** 2 for x in label_rescale[:, para])
                r2_mean[para] = 1 - dividend / divisor
                #calculate nrmse
                nrmse[para]=np.sqrt(np.mean((mean[:,para]-label_rescale[:,para])**2))/(np.max(label_rescale[:,para])-np.min(label_rescale[:,para]))
            
        np.savez(self.output_dir+output_name,label=label_rescale ,mean=mean,lower=lower,
                 upper=upper,rank=rank,cal_err=cal_err,r2_mean=r2_mean,nrmse=nrmse)
        logging.info('Saved statistics')
        separator()
        
    def plot_calibration(self) -> None:
        """
        Plot calibration curves.

        This method generates calibration plots based on the statistics stored in the 'statistics.npz' file.
        The plots show the posterior mean and the 68% confidence interval for each parameter, along with the true values.

        Returns:
            None
        """
        logging.info('Make calibration plots')
        data = np.load(self.output_dir+'statistics.npz')
        label = data['label']
        mean = data['mean']
        lower = data['lower']
        upper = data['upper']
        error_low = np.abs(mean-lower)
        error_up = np.abs(upper-mean)
        size = self.params['fontsize']
        with PdfPages(self.output_dir+'calibration.pdf') as pdf:
            for para in range(len(self.parameters)):
                plt.figure(figsize=(9,6))
                plt.errorbar(label[:,para],mean[:,para],yerr=(error_low[:,para],error_up[:,para])
                        ,fmt='.',markersize=3,color='darkred',alpha=1,lw=1., ecolor='lightsteelblue')
                plt.plot(label[:,para],label[:,para],color='k',zorder=100,lw=1)
                plt.title(self.parameters[para][3],fontsize=size)
                plt.xlabel('True',fontsize=size)
                plt.ylabel('Posterior (68% CL)',fontsize=size)
                plt.xticks(fontsize=size)
                plt.yticks(fontsize=size)
                if para == 4:
                    plt.ylim(3.9,None)
                pdf.savefig()
                plt.close()
        separator()
        
    def plot_rank_statistic(self) -> None:
        """
        Plot the rank statistic.

        This method generates rank statistic plots based on the data stored in the 'statistics.npz' file.
        It saves the plots as a PDF file named 'rank_statistic.pdf' in the output directory. It can be used
        to detect visual biases.

        Returns:
            None
        """
        logging.info('Make rank statistic plots')
        data = np.load(self.output_dir+'statistics.npz')
        label = data['label']
        rank = data['rank']
        size = self.params['fontsize']
        bins = 15
        sample_size = int(np.max(rank))
        ranges = np.linspace(-500, sample_size+500, bins)
        avg = label.shape[0]/bins
        low, up = binom.interval(0.99,label.shape[0],1/bins)
        with PdfPages(self.output_dir+'rank_statistic.pdf') as pdf:
            for para in range(len(self.parameters)):
                plt.figure(figsize=(9,6))
                plt.hist(rank[:,para],bins=bins ,ec='teal',histtype=u'step')
                plt.fill_between(ranges,low, up,color='k',alpha=0.2)
                plt.plot(ranges,np.ones(bins)*avg,color='k')
                plt.title(self.parameters[para][3],fontsize=size)
                plt.xlabel('Rank statistic',fontsize=size)
                plt.xticks(fontsize=size)
                plt.yticks(fontsize=0)
                pdf.savefig()
                plt.close()
        separator()
        
    def sample_fiducial(self, index: int, sample_size: int = 10000) -> Tuple[np.array, np.ndarray]:
        """
        Generate samples from the posterior for a given fiducial dataset. The dataset is specified by its index.

        Args:
            index (int): The index of the sample.
            sample_size (int, optional): The number of samples to generate. Defaults to 10000.

        Returns:
            Tuple[np.array, np.ndarray]: A tuple containing the rescaled label and generated samples.
        """
        # check if the CNN output is already loaded
        try:
            label = self.label[index]
            cnn_pred = self.cnn_pred[index]
        except:
            path = self.test_paths[index]
            X, y = self.data_reader([path], self.device)
            pred = self.cnn(X)
            cnn_pred = pred.detach().cpu().numpy()
            label = y.detach().cpu().numpy()
            
        z = torch.randn((sample_size,len(self.parameters))).to(self.device)
        z = torch.randn((sample_size,len(self.parameters))).to(self.device)
        samples, _ = self.flow(z, c = [torch.Tensor(cnn_pred).repeat((sample_size,1)).to(self.device)], rev=True)
        samples = samples.detach().cpu().numpy()
        samples = self.rescale(samples)
        return self.rescale(label.reshape(1,-1))[0], samples
        
        
    def plot_corner(self, label: np.array, samples: np.ndarray) -> Figure:
        """
        Plots a corner plot with the ground truth.

        Args:
            label (np.array): The truth for each parameter.
            samples (np.ndarray): The samples for each parameter.

        Returns:
            Figure: The generated corner plot figure.
        """
        size = self.params['fontsize']
        lim=[]
        for i in range(len(self.parameters)):
            lim.append([self.parameters[i][l+1] for l in range(2)])
               #make getdist plot
        names=[self.parameters[para][3] for para in range(len(self.parameters))]
        lab=[self.parameters[para][4] for para in range(len(self.parameters))]
        samp_mc = MCSamples(samples=samples, names=names, labels=lab)  
        g = plots.get_subplot_plotter()
        g.settings.legend_fontsize = size
        g.settings.axes_fontsize = size
        g.settings.axes_labelsize = size
        g.settings.linewidth = 2
        g.settings.line_labels = False
        color = [self.params['corner_plot']['color']]
        g.triangle_plot([samp_mc],filled=True,legend_loc='upper right',colors=color,contour_colors=color)
        for i in range(6):
            ax = g.subplots[i,i].axes
            ax.axvline(label[i], color='k', ls='--',lw=2)
        n=0
        m=1
        while n<5:
            ax = g.subplots[m,n].axes
            ax.scatter(label[n], label[m], color='k', marker='x', s=100)
            m+=1
            if m==6:
                n+=1
                m=n+1
        fig = g.fig
        post_patch = mpatches.Patch(color=color[0], label='Posterior')
        true_line = mlines.Line2D([], [], color='k', marker='x',ls='--',lw=2,
                              markersize=10, label='True')
        fig.legend(handles=[post_patch,true_line],bbox_to_anchor=(0.98, 0.98),fontsize=size)
        return fig
    
    def plot_many_corners(self) -> None:
        """
        Plot multiple corner plots based on the specified parameters.
        How many plots are generated and which lightcones are used is determined by the parameters in the yaml file.
        Options are to use a random lightcone, the default lightcone, or a list of indices.

        Returns:
            None
        """
        fiducial = self.params['corner_plot']['fiducial']
        if fiducial == 'random':
            n_plots = self.params['corner_plot']['n_plots']
            index = []
            for i in range(n_plots):
                index.append(np.random.randint(self.num_lightcones))
        elif fiducial == 'default':
            index = [self.name_to_index(direction='forward', name=4692)]
        elif type(fiducial) == list:
            index = []
            for f in fiducial:
                index.append(self.name_to_index(direction='forward', name=f))
        else:
            raise ValueError('Fiducial must be random, default or a list of indices')
        figures = []
        os.makedirs(self.output_dir+'fiducial_samples', exist_ok=True)
        logging.info(f'Make {len(index)} corner plots')
        for i in index:
            label, samples = self.sample_fiducial(i)
            name = self.name_to_index(direction='backward', index=i)
            np.savez(self.output_dir+f'fiducial_samples/fiducial_{name}.npz',label=label,samples=samples)
            logging.info(f'Inference for LC no.{name}')
            figures.append(self.plot_corner(label, samples))
        with PdfPages(self.output_dir+f'corner.pdf') as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
        separator()
    
    def check_latent(self) -> None:
        """
        Make latent space plots.

        This method generates a plot of the latent distribution and compares it to 
        the desired Gaussian.

        Returns:
            None
        """
        logging.info('Make latent space plots')
        size = self.params['fontsize']
        try:
            label_t = torch.tensor(self.label.astype('float32')).to(self.device)
            pred_t = torch.Tensor(self.cnn_pred.astype('float32')).to(self.device)
        except:
            self.cnn_pred, self.label = self.find_cnn_output()
            label_t = torch.tensor(self.label.astype('float32')).to(self.device)
            pred_t = torch.Tensor(self.cnn_pred.astype('float32')).to(self.device)
        z, _ = self.flow(label_t, c=[pred_t])
        samp = z.detach().cpu().numpy()
        names = ["z_%s" % i for i in range(len(self.parameters))]
        lab = ["z_%s" % i for i in range(len(self.parameters))]
        r = np.random.randn(100000, 6)
        samp_gaussian = MCSamples(samples=r, names=names, labels=lab)
        samp_mc = MCSamples(samples=samp, names=names, labels=lab)
        g = plots.get_subplot_plotter()
        g.settings.legend_fontsize = size
        g.settings.axes_fontsize = size
        g.settings.axes_labelsize = size
        g.settings.linewidth = 2
        color = ['darkred', 'k']
        g.triangle_plot([samp_mc, samp_gaussian], filled=[True, False],
                        legend_labels=['Latent Distribution', 'Gaussian'],
                        legend_loc='upper right', colors=color, contour_colors=color)
        plt.savefig(self.output_dir + 'latent.pdf')
        plt.close()
        
    def main(self) -> None:
        """
        Executes the main plotting functionality based on the specified parameters.

        This method performs the following actions:
        - Calculates statistics if specified in the parameters.
        - Plots calibration if specified in the parameters.
        - Plots rank statistic if specified in the parameters.
        - Plots corner plots if specified in the parameters.
        - Checks latent variables if specified in the parameters.

        Returns:
            None
        """
        if self.params['calc_statistics']['do_it']:
            try: output_name = self.params['calc_statistics']['output_name']
            except: output_name = 'statistics.npz'
            self.calc_statistics(output_name, 
                                 sample_size=self.params['calc_statistics']['sample_size'],
                                 cal_error=self.params['calc_statistics']['cal_error'],
                                 rank_stat=self.params['calc_statistics']['rank_stat'],
                                 mean_metrics=self.params['calc_statistics']['mean_metrics'])
        if self.params['plot_calibration']:
            self.plot_calibration()
        if self.params['plot_rank_stat']:
            self.plot_rank_statistic()
        if self.params['corner_plot']:
            self.plot_many_corners()
        if self.params['check_latent']:
            self.check_latent()
        separator()
        logging.info('Done plotting')
        separator()
        
    
        
            
        
                
        
        
        
        
                
                
                
                
        
        
        
        

        
    
    