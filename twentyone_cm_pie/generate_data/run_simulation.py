import os
import glob
import sys
import logging
import shutil
import math
import numpy as np
from typing import Tuple

import py21cmfast as p21c

from ..util.logger import separator

class Simulation:
    """
    Simulation class for creating, managing, and analyzing cosmological simulations using 21cmFAST.
    
    Attributes:
        params (dict): Configuration parameters for the simulation.
        destination (str): Directory where simulation results are saved.
        destination_noise (str): Directory where noise-added simulation results are saved.
        cache_dest (str): Directory for caching intermediate simulation results.
    """
    
    def __init__(self, params: dict) -> None:
        """
        Initialize the Simulation object with the provided parameters.
        
        Args:
            params (dict): Configuration parameters for the simulation.
        """
        self.params = params
        self.destination = params['create']['destination']
        os.makedirs(self.destination, exist_ok=True)
        if self.params['noise']['add']:
            self.destination_noise = self.params['noise']['destination']
            os.makedirs(self.destination_noise, exist_ok=True)
        separator()
    
    def find_name(self) -> str:
        """
        Find a unique name for the output simulation file that does not already exist and create a cache directory.
        
        Returns:
            str: Unique filename for the output simulation.
        """
        i = 1
        filename = self.destination+'/run'+str(i)+'.npz'
        while(filename in glob.glob(self.destination+'/*')):
            i += np.random.randint(1, 10000)
            filename = self.destination+'/run'+str(i)+'.npz'
        self.cache_dest = '.cache/_cache'+str(i)
        os.makedirs(self.cache_dest, exist_ok=True)
        return filename
       
    def sample_parameters(self) -> np.array:
        """
        Sample parameters for the simulation based on the provided configuration.
        
        Returns:
            np.array: Array of sampled parameters.
        """
        if self.params['create']['param_sampling'] == 'random':
            WDM = np.random.uniform(0.3, 10.0)
            OMm = np.random.uniform(0.2, 0.4)
            E0 = np.random.uniform(100, 1500)
            LX = np.random.uniform(38, 42)
            Tvir = np.random.uniform(4, 5.3)
            Zeta = np.random.uniform(10, 250)
        elif self.params['create']['param_sampling'] == 'default':
            WDM = 3.5
            OMm = 0.3158
            E0 = 500.0
            LX = 40.0
            Tvir = 4.69897
            Zeta = 30.0
            
        return np.array([WDM, OMm, E0, LX, Tvir, Zeta])
    
    def run_sim(self, parameters: np.array) -> p21c.LightCone:
        """
        Run the simulation with the given parameters.
        
        Args:
            parameters (np.array): Array of parameters to use in the simulation.
        
        Returns:
            p21c.LightCone: The simulated lightcone object.
        """
        HEIGHT_DIM = 140
        BOX_LEN = 200
        Z_MIN = 5.0
        N_THREAD = self.params['create']['threads']
        WDM, OMm, E0, LX, Tvir, Zeta = parameters
        logging.info(f'Running simulation with parameters: {parameters}')
        p21c.inputs.global_params.M_WDM = WDM
        lightcone = p21c.run_lightcone(
            redshift=Z_MIN,
            cosmo_params=p21c.CosmoParams(OMm=OMm),
            astro_params=p21c.AstroParams(HII_EFF_FACTOR=Zeta, L_X=LX, NU_X_THRESH=E0, ION_Tvir_MIN=Tvir),
            user_params={"HII_DIM": HEIGHT_DIM, "BOX_LEN": BOX_LEN, "PERTURB_ON_HIGH_RES": True, "N_THREADS": N_THREAD, "USE_INTERPOLATION_TABLES": True},
            flag_options={"USE_TS_FLUCT": True, "INHOMO_RECO": True},
            direc=self.cache_dest,
        )
        return lightcone
        
    def filter_limits(self, lightcone: p21c.LightCone) -> bool:
        """
        Check if tau and global xH are within acceptable limits.
        
        Args:
            lightcone (p21c.LightCone): The simulated lightcone object.
        
        Returns:
            bool: True if tau and global xH are within limits, False otherwise.
        """
        with open("21cm_pie/generate_data/redshifts5.npy", "rb") as data:
            redshifts = list(np.load(data, allow_pickle=True))
        redshifts.sort()
        gxH = lightcone.global_xH
        gxH = gxH[::-1]
        tau = p21c.compute_tau(redshifts=redshifts, global_xHI=gxH)
        return tau <= 0.089 and gxH[0] <= 0.1
        
    def save(self, filename: str, brightness_temp: np.ndarray, parameters: np.array) -> None:
        """
        Save the brightness temperature and parameters to a file and remove the cache directory.
        
        Args:
            filename (str): The filename to save the results.
            brightness_temp (np.ndarray): The brightness temperature data.
            parameters (np.array): The simulation parameters.
        """
        np.savez(filename, image=brightness_temp, label=parameters)
        try:
            shutil.rmtree(self.cache_dest)
        except:
            pass
    
    def read_noise_files(self) -> list:
        """
        Read noise files based on the noise level specified in the parameters.
        
        Returns:
            list: List of filenames for the noise data.
        """
        noise_level = self.params['noise']['level']
        if noise_level == "opt":
            files = glob.glob("21cm_pie/generate_data/calcfiles/opt_mocks/SKA1_Lowtrack_6.0hr_opt_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz")
        elif noise_level == "mod":
            files = glob.glob("21cm_pie/generate_data/calcfiles/mod_mocks/SKA1_Lowtrack_6.0hr_mod_0.*_LargeHII_Pk_Ts1_Tb9_nf0.52_v2.npz")
        else:
            logging.info("Please choose a valid foreground model")
            sys.exit()
        files.sort(reverse=True)
        return files
         
    def add_noise(self, brightness_temp: np.ndarray, parameters: np.array) -> np.ndarray:
        """
        Add noise to the simulation.
        
        Args:
            brightness_temp (np.ndarray): The brightness temperature data.
            parameters (np.array): The simulation parameters.
        
        Returns:
            np.ndarray: The brightness temperature data with added noise.
        """
        logging.info('Create mock')
        with open("21cm_pie/generate_data/redshifts5.npy", "rb") as data:
            box_redshifts = list(np.load(data, allow_pickle=True))
            box_redshifts.sort()
        cosmo_params = p21c.CosmoParams(OMm=parameters[1])
        astro_params = p21c.AstroParams(INHOMO_RECO=True)
        user_params = p21c.UserParams(HII_DIM=140, BOX_LEN=200)
        flag_options = p21c.FlagOptions()
        sim_lightcone = p21c.LightCone(5., user_params, cosmo_params, astro_params, flag_options, 0,
                                       {"brightness_temp": brightness_temp}, 35.05)
        redshifts = sim_lightcone.lightcone_redshifts
        box_len = np.array([])
        y = 0
        z = 0
        for x in range(len(brightness_temp[0][0])):
            if redshifts[x] > (box_redshifts[y + 1] + box_redshifts[y]) / 2:
                box_len = np.append(box_len, x - z)
                y += 1
                z = x
        box_len = np.append(box_len, x - z + 1)
        y = 0
        delta_T_split = []
        for x in box_len:
            delta_T_split.append(brightness_temp[:,:,int(y):int(x+y)])
            y+=x
            
        mock_lc = np.zeros(brightness_temp.shape)
        cell_size = 200 / 140
        hii_dim = 140
        k140 = np.fft.fftfreq(140, d=cell_size / 2. / np.pi)
        index1 = 0
        index2 = 0
        files = self.read_noise_files()
        for x in range(len(box_len)):
            with np.load(files[x]) as data:
                ks = data["ks"]
                T_errs = data["T_errs"]
            kbox = np.fft.rfftfreq(int(box_len[x]), d=cell_size / 2. / np.pi)
            volume = hii_dim * hii_dim * box_len[x] * cell_size ** 3
            err21a = np.random.normal(loc=0.0, scale=1.0, size=(hii_dim, hii_dim, int(box_len[x])))
            err21b = np.random.normal(loc=0.0, scale=1.0, size=(hii_dim, hii_dim, int(box_len[x])))
            deldel_T = np.fft.rfftn(delta_T_split[x], s=(hii_dim, hii_dim, int(box_len[x])))
            deldel_T_noise = np.zeros((hii_dim, hii_dim, int(box_len[x])), dtype=np.complex_)
            deldel_T_mock = np.zeros((hii_dim, hii_dim, int(box_len[x])), dtype=np.complex_)
            
            for n_x in range(hii_dim):
                for n_y in range(hii_dim):
                    for n_z in range(int(box_len[x] / 2 + 1)):
                        k_mag = math.sqrt(k140[n_x] ** 2 + k140[n_y] ** 2 + kbox[n_z] ** 2)
                        err21 = np.interp(k_mag, ks, T_errs)
                        
                        if k_mag:
                            deldel_T_noise[n_x, n_y, n_z] = math.sqrt(math.pi * math.pi * volume / k_mag ** 3 * err21) * (err21a[n_x, n_y, n_z] + err21b[n_x, n_y, n_z] * 1j)
                        else:
                            deldel_T_noise[n_x, n_y, n_z] = 0
                        
                        if err21 >= 1000:
                            deldel_T_mock[n_x, n_y, n_z] = 0
                        else:
                            deldel_T_mock[n_x, n_y, n_z] = deldel_T[n_x, n_y, n_z] + deldel_T_noise[n_x, n_y, n_z] / cell_size ** 3
            
            delta_T_mock = np.fft.irfftn(deldel_T_mock, s=(hii_dim, hii_dim, box_len[x]))
            index1 = index2
            index2 += delta_T_mock.shape[2]
            mock_lc[:, :, index1:index2] = delta_T_mock
            if x % 5 == 0:
                logging.info(f'mock created to {int(100 * index2 / 2350)}%')
        return mock_lc
    
    def create_lightcones(self) -> None:
        """
        Create multiple lightcone simulations based on the parameters.
        """
        logging.info('Start creating Simulations')
        N = self.params['create']['n_sims']
        j = 0
        while j < N:
            logging.info(f'Creating lightcone {j + 1}/{N}')
            filename = self.find_name()
            logging.info(f'Saving to {filename}')
            parameters = self.sample_parameters()
            lightcone = self.run_sim(parameters)
            if self.filter_limits(lightcone):
                attr = getattr(lightcone, "brightness_temp")
                brightness_temp = attr[:, :, :2350]
                self.save(filename, brightness_temp, parameters)
                j += 1
            else:
                logging.info('Simulation rejected')
                continue
            if self.params['noise']['add']:
                mock_lc = self.add_noise(brightness_temp, parameters)
                run_name = filename.split('/')[-1]
                self.save(self.destination_noise + '/' + run_name, mock_lc, parameters)
        logging.info('All lightcones created')
        
    def read_lightcones(self, filename: str) -> Tuple[np.ndarray, np.array]:
        """
        Read the lightcone data from a file.
        
        Args:
            filename (str): The filename of the lightcone data.
        
        Returns:
            Tuple[np.ndarray, np.array]: The brightness temperature data and the parameters.
        """
        cone = np.load(filename)
        brightness_temp = cone['image']
        label = cone['label']
        return brightness_temp, label
        
    def convert_lightcones(self) -> None:
        """
        Convert existing lightcone simulations to add noise.
        """
        params = self.params['convert']
        files = glob.glob(params['source'] + '*')
        if params['n_sims'] == 'all':
            logging.info(f'Converting all ({len(files)}) lightcones')
            for file in files:
                brightness_temp, label = self.read_lightcones(file)
                self.add_noise(brightness_temp, label)
                save_name = file.split('/')[-1]
                self.save(self.destination_noise + '/' + save_name, brightness_temp, label)
        else:
            sys.exit()
    
    def main(self) -> None:
        """
        Main method to create and/or convert lightcones based on the parameters.
        """
        if self.params['create']['do_it']:
            self.create_lightcones()
        if self.params['convert']['do_it']:
            self.convert_lightcones()
        separator()
        logging.info('All tasks completed')
        separator()
