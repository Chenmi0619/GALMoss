import numpy as np
import torch
from itertools import chain
import pandas as pd
from tqdm import tqdm
import torch_optimizer as optim
import os
import h5py
import torch.utils.data as Data
import torch.nn.functional as F
from astropy.io import fits
from astropy.table import Table
from typing import Union

from galmoss.profile.light_profile import LightProfile
from galmoss.data import DataSet
from galmoss.uncertainty import Uncertainty

     
class FittingRepo():
    """
    This class owns functions that relates to fitting metrics initializing, 
    updating and storaging in fitting process. 
    """
    
    def initialize_fitting_metrics(self):
        """
        Initial the matrix stores fitting metrics.
        
        Returns
        -------
        stop_update_count
            The number of galaxies that already stop fitting.
        batched_eff_iter
            The iteration being used when the belonging galaxy stop fitting 
            when reaching the updating shreshold in this batch.
        batched_best_chi_mu
            The best chi_mu value of belonging galaxy in this batch. 
        """
        stop_update_count = torch.zeros((len(self._index))).to(self.dataset.device)
        batched_eff_iter = torch.ones((len(self._index))).to(self.dataset.device)
        batched_best_chi_mu = torch.empty((0,)).to(self.dataset.device)
        return (stop_update_count, batched_eff_iter, batched_best_chi_mu)
         
    # def update_best_chi_mu(self, 
    #                         chi_mu: torch.tensor, 
    #                         index: Union[torch.tensor, bool] = None):
    #     """
    #     Update the best chi_mu value during the optimization process. Not 
    #     every iteration in the optimization process results in a better 
    #     outcome with a lower chi-square. Therefore, updates are made only 
    #     in iterations that successfully reduce the chi-square, guided by 
    #     the given index.

    #     Parameters
    #     ----------
    #     chi_mu
    #         The chi_mu being calculated in one fitting iteration.
    #     index
    #         Specifies the position in the best chi_mu value matrix to be
    #         updated. If None, updated the entire matrix.
    #     """            
    #     if index == None:
    #         self.batched_best_chi_mu = chi_mu
    #     else:
    #         self.batched_best_chi_mu[index] = chi_mu[index]
                    
    def fitting_metrics_append(self, batched_eff_iter, batched_best_chi_mu):
        """
        Append the fitting_metrics for each batch of galaxies after
        the completion of each batch fitting process.

        Parameters
        ----------
        batched_eff_iter
            The iteration being used when the belonging galaxy stop fitting 
            when reaching the updating shreshold in this batch.
        batched_best_chi_mu
            The best chi_mu value of belonging galaxy in this batch. 
        """   
        if hasattr(self, 'total_best_chi_mu'):
            self.total_best_chi_mu = torch.concat(
                (self.total_best_chi_mu, batched_best_chi_mu), dim=0)        
        else:
            self.total_best_chi_mu = batched_best_chi_mu
        
        if hasattr(self, 'total_eff_iter'):
            self.total_eff_iter = torch.concat(
                (self.total_eff_iter, batched_eff_iter), dim=0)        
        else:
            self.total_eff_iter = batched_eff_iter

    @property
    def chi_mu_numpy(self) -> np.ndarray:
        """
        Turns the chi_mu data after fitting into numpy.

        Returns
        -------
        np.ndarray
            The chi_mu values as a NumPy array.
        """
        return self._chi_mu.detach().cpu().numpy()

    def refresh_variable(self, P_dict, optim_list):
        """
        This function is for refresh the stepped value inside optimizer to
        the parameters instance, in sequence of the parameter diction 
        being given.

        Parameters
        ----------
        P_dict:
            The dictionary of variable parameters.
        optim_list:
            The optim list stores the stepped value of such parameters.
        """
        for ams, param in enumerate(P_dict.values()):
            data = optim_list[ams]['params'][0]
            param.update_updating_val(data)
 
    def load_constant(self, P_dict, P_value):
        """
        This function is for set the fixed parameters (constant) before 
        every batch-size of training.

        Parameters
        ----------
        P_dict:
            The dictionary of fixed parameters.
        P_value:
            The total value matrix stores the initial mapped value of such
            parameters.
        """        
        ams = 0
        for param in P_dict.values():
            param.update_updating_val(P_value[:, ams: ams+param.dim2_length])
            param.update_best_val(index=None)    
            ams += param.dim2_length    
  
    def update_param_best_value(self, index):
        """
        Update the best value of each parameters during the optimization
        process. 
        
        Parameters
        ----------
        index
            Specifies the position in the best value matrix to be
            refreshed. If None, refreshes the entire matrix.
        """          
        with torch.no_grad(): 
            for param in self.dataset.Profiles.variableParam.values():
                param.update_best_val(index)
                param.grad = 0
        
    def append_param_best_value(self):
        """
        Append the best fitted values of each parameters after the 
        completion of each batch fitting process.
        """ 
        for param in self.dataset.Profiles.variableParam.values():
            param.best_value_append(param.batched_mapped_best_value)
            
    def save_param(self): 
        """
        Save the best fitted values of each parameters after the total 
        fitting process ends. A dictionary will be used to store the key
        ("{}_{}_(varaible/fixed)".format(profile_name, param_name)) and 
        the corresponding fitted value.
        
        If the mode is not fitting, this function will save the 
        uncertainty together, with the key:
            ("{}_{}_err".format(profile_name, param_name))
        and the corresponding uncertainty value.
        
        After the dictionary is defined, the file will be saved follows 
        the attributes `result_type`.
        """         
        data = {}
        data["galaxy_idx"] = self.dataset.galaxy_index
        data["chi_mu"] = self.total_best_chi_mu.detach().cpu().numpy()
        for profile_name, profile in self.dataset.Profiles.lightProfile_dict.items():          
            for variable_param in profile.variable_dic.values():
                key = "{}_{}_(varaible)".format(
                    profile_name, 
                    variable_param.param_name
                    )
                if variable_param.dim == "1d":
                    data[key] = (variable_param.total_best_value_numpy
                                 .reshape(variable_param.param_length))
                if ((self.dataset.mode in ["bootstrap", "covar_mat"])
                    and (variable_param.fit)):
                    key = "{}_{}_err".format(
                        profile_name, 
                        variable_param.param_name
                        )
                    data[key] = variable_param.total_uncertainty_numpy  
            for fixed_param in profile.fixed_dic.values():
                key = "{}_{}_(fixed)".format(
                    profile_name, 
                    fixed_param.param_name
                    )
                data[key] = (fixed_param.total_initial_value_numpy
                                 .reshape(fixed_param.param_length)) 
        if self.dataset.result_type == "FITS":              
            hdu = fits.BinTableHDU(Table(data))
            hdu.writeto(os.path.join(self.dataset.result_path, 'result.fits'), 
                        overwrite=True) 
        elif self.dataset.result_type == "CSV":
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.dataset.result_path, 'result.csv'), 
                      index=False)   
        else:
            with h5py.File(os.path.join(self.dataset.result_path, 
                                        'result.hdf5'), 'w') as hdf:
                for key, value in data.items():
                    hdf.create_dataset(key, data=value)              

    def save_image(self):
        """
        This method saves the img block as FITS file, which includes galaxy 
        image, model image, residual image and the image of sub-components. 
        Each sub-components will be convoluted either.
        """                 
        conv_model = self.modeling(
            profile_dict=self.dataset.Profiles.conv_profile_dict, 
            mode="best_model")
        no_conv_model = self.modeling(
            profile_dict=self.dataset.Profiles.no_conv_profile_dict, 
            mode="best_model")
        component_cube = torch.cat((conv_model, no_conv_model), dim=0)
                
        if self.dataset.psf_path is not None:
            model_tot = (self.add_convolution(conv_model.sum(axis=0)) 
                           + no_conv_model.sum(axis=0))
            residual = (self.data_cube[:, 0] - model_tot).detach().cpu().numpy()
            for profile_idx in np.arange(
                len(self.dataset.Profiles.conv_profile_dict)
                ):
                conv_model[profile_idx] = torch.squeeze(
                    F.conv2d(
                        conv_model[profile_idx].unsqueeze(0), 
                        self._psf.unsqueeze(1), 
                        padding="same", 
                        groups=len(self._index)
                        )
                ) 
        else:
            model_tot = conv_model.sum(axis=0) + no_conv_model.sum(axis=0)
            residual = (self.data_cube[:, 0] - model_tot).detach().cpu().numpy()
            for profile_idx in np.arange(
                len(self.dataset.Profiles.conv_profile_dict)
                ):
                conv_model[profile_idx] = conv_model[profile_idx]
                                     
        galaxy_data = self.data_cube[:, 0].detach().cpu().numpy()
        model_tot = model_tot.detach().cpu().numpy()
        for ams, galaxy in enumerate(self._index):
            primary_hdu = fits.PrimaryHDU()
            image_hdu = fits.ImageHDU(galaxy_data[ams])
            image_hdu.header['KEYWORD'] = 'data'

            model_hdu = fits.ImageHDU(model_tot[ams])
            model_hdu.header['KEYWORD'] = 'model'
                
            residual_hdu = fits.ImageHDU(residual[ams])
            residual_hdu.header['KEYWORD'] = 'residual'

            component_hdulist = [primary_hdu, image_hdu, model_hdu, residual_hdu]

            for abs, (key, key_value) in enumerate(
                chain(self.dataset.Profiles.conv_profile_dict.items(),
                self.dataset.Profiles.no_conv_profile_dict.items())
                ):  
                if key_value.saving:     
                    component_hdu = fits.ImageHDU(
                        component_cube[abs, ams].detach().cpu().numpy())
                    component_hdu.header['KEYWORD'] = key
                    component_hdulist.append(component_hdu)      
        
            component_hdulist = fits.HDUList(component_hdulist)
            component_hdulist.writeto(
                os.path.join(self.dataset.img_block_path, galaxy + '.fits'),
                overwrite=True
                )

          
class Fitting(FittingRepo, Uncertainty):
    """
    This class owns functions that relates to fitting process. 
    
    Parameters
    ----------
    dataset
        The object instanced from `DataSet` class, used for Load the 
        images and Parameters in parallel.
    batch_size
        The number of galaxies being fitted together in one batch. This
        number should as big as possible until the memory usage and GPU
        usage reach the limit.    
    iteration
        The number of fitting iteration for the fitting process, the 
        recommand number is 1,000 ~ 1,500.
    optimizer
        The type of optimizer. Galmoss supports any kind of optimizers in 
        PyTorch. The defalt one is DiffGrad.
    threshold & early_stop
        The galaxy which 
        (best_chi_mu  - chi_mu) / best_chi_mu  < `threshold` 
        more than `early_stop` time will stop fitting. The defalt number is
        10e-5 and 10.
    """
    def __init__(
        self, 
        dataset: DataSet,
        batch_size: int,
        iteration: int,
        optimizer = optim.DiffGrad,
        threshold = 1e-5,
        early_stop = 50
    ):
        self.dataset = dataset
        self.optimizer = optimizer
        
        if isinstance(batch_size, int) and batch_size > 0:
            self.batch_size = batch_size
        else:
            raise ValueError("batch size must be an integer greater than 0")

        if isinstance(iteration, int) and iteration > 0:
            self.iteration = iteration
        else:
            raise ValueError("iteration must be an integer greater than 0")

        if isinstance(threshold, float) and threshold > 0:
            self.threshold = threshold
        else:
            raise ValueError("threshold must be a float greater than 0")  

        if isinstance(early_stop, int) and early_stop > 0:
            self.early_stop = early_stop
        else:
            raise ValueError("early_stop must be an integer greater than 0")
                       
        super().__init__()  

    def make_optim_list(self, P_dict, P_value): 
        """
        Make the dictionary for being load in the optimizer.
        
        Parameters
        ----------
        P_dict
            The dictionary of parameter objects.
        P_value
            The total value matrix for the parameters inside P_dict.

        Returns
        -------
        optim_list
            The optimizer list, which includes dictionaries for each 
            parameters, which contains values and learning rate.
        """          
        optim_list = []
        ams = 0
        for param in P_dict.values():        
            param_dict = {}
            param_value = P_value[:, ams: ams+param.dim2_length].clone()
            param_dict['params'] = param_value.detach().requires_grad_(True)
            param_dict['lr'] = param.step_length
            optim_list.append(param_dict)
            ams += param.dim2_length
        return optim_list
                  
    def modeling(self, profile_dict, mode="updating_model"):
        """
        Generate the total model image matrix.
        
        Parameters
        ----------
        profile_dict
            The dictionary of profile objects.
        mode
            The string representing which value to choose from the 
            parameters.

        Returns
        -------
        model_matrix
            The total model matrix, with shape 
            (profile_num, galaxy_num, n, m).
            (n, m) is the data shape.
        """            
        model_repo = []
        for profile in profile_dict.values():
            model_repo.append(profile.image_via_grid_from(self.grid, mode))
        if model_repo:
            model_matrix = torch.stack(model_repo) 
        else:
            model_matrix = torch.zeros((0)).to(self.dataset.device)
        return model_matrix
      
    def make_grid(self):
        """
        Generate the grid for fitting process.
        """             
        x = torch.linspace(0.5, 
                           self.dataset.data_size[1]-0.5, 
                           self.dataset.data_size[1], 
                           dtype=self.dataset.data_type).to(self.dataset.device)
        y = torch.linspace(self.dataset.data_size[0]-0.5, 
                           0.5, 
                           self.dataset.data_size[0], 
                           dtype=self.dataset.data_type).to(self.dataset.device)
        xy = torch.meshgrid(y, x, indexing='ij')      
        self.grid = xy
          
    def revise_mask(self):
        """
        To generage the mask image follows various mode.
        If the fitting process already loads mask image, then the original 
        mask image will be call from `cls.data_cube`. Otherwise, it will 
        be a tensor filled with the scalar value 1, with the shape same as 
        the galaxy data image (In practical, the calculation will multiply 1 
        instead). The original mask image will be used In the fitting mode. 
        
        In bootstrap mode, the original mask data will be resampled.
        
        Returns
        -------
        mask
            The mask image that will be used in chi-square calculations.
        mask_num
            The effective pixel number of mask image, which will be used in
            freedom degree calculations.
        """

        if "seg" in self.dataset.data_cube_idx:
            if self.dataset.mode == "bootstrap":
                mask = self.resample_mask(
                    self.data_cube[:, 
                                   self.dataset.data_cube_idx.index("seg")
                                   ]
                    )
                mask_num = torch.sum(mask, dim=[1, 2])
            else:
                mask = self.data_cube[:, 
                                      self.dataset.data_cube_idx.index("seg")
                                      ]
                mask_num = torch.sum(mask, dim=[1, 2])                
        else:
            if self.dataset.mode == "bootstrap":
                mask = self.resample_mask(
                    torch.ones(
                        self.data_cube[:, 
                                       self.dataset.data_cube_idx.index("data")
                                       ].shape
                    ).to(self.dataset.device)
                )
                mask_num = torch.sum(mask, dim=[1, 2])
            else:
                mask = 1
                mask_num = self.dataset.data_size[0] * self.dataset.data_size[1]
        return mask, mask_num

    def add_convolution(self, to_conv_model):
        """
        Make the convolution to the input model image with the
        psf images.

        Parameters
        ----------
        to_conv_model
            The images need to be convolved.

        Returns
        -------
            The convolved images.
        """      
        if (self.dataset.psf_path is not None) and (len(to_conv_model) >0):
            noramlize_factor = torch.sum(
                self._psf, 
                dim=[1, 2]
                ).unsqueeze(1).unsqueeze(1)
            conved_model = torch.squeeze(F.conv2d(
                to_conv_model.unsqueeze(0), 
                self._psf.unsqueeze(1), 
                padding="same", 
                groups=len(self._index))) / noramlize_factor        
        else:
            conved_model = to_conv_model
        return conved_model
       
    def detect_update(self, fitting_metrics, chi_mu, iter):
        """
        Not every iteration in the optimization process results in a 
        better outcome with a lower chi-square. Therefore, updates are 
        made only in iterations that successfully reduce the chi-square, 
        which are detected here in `is_better`.

        `improvement_small` is galaxies their Δchi_mu is smaller than the 
        thresholds. if `early_stop` number of iterations achive this statement 
        continuously, this galaxy will be marked as early-stop galaxy.
        At present, only all the galaxies are early-stoped galaxy, the fitting 
        process will finish.
        
        Parameters
        ----------
        fitting_metrics
            A tuple includes stop_update_count, batched_eff_iter and
            batched_best_chi_mu. `stop_update_count` represents the
            number of iterations achive `improvement_small` statement 
            continuously. `batched_eff_iter` represents how much 
            iterations each galaxy used to achive the best value.
            `batched_best_chi_mu` represents the best chi_mu.
        chi_mu
            The chi_mu being calculated in this iteration.
        iter
            The value of this iteration.
            
        Returns
        -------
        update
            If True, continue fitting, otherwise break the loop.
        fitting_metrics 
            The updated fitting_metrics.
        """          
        (stop_update_count, 
         batched_eff_iter, 
         batched_best_chi_mu) = fitting_metrics
        
        if iter == 0:
            batched_best_chi_mu = chi_mu
            stop_update_count += 1
            self.update_param_best_value(index=None)
           
        else: 
            is_better = chi_mu < batched_best_chi_mu
            improvement_small = ((torch.abs(chi_mu - 1) 
                                 < torch.abs(batched_best_chi_mu - 1)) 
                                 & ((batched_best_chi_mu - chi_mu) 
                                    / batched_best_chi_mu  < self.threshold))
            
            batched_eff_iter[is_better] = iter
            batched_best_chi_mu[is_better] = chi_mu[is_better]            
            stop_update_count[~improvement_small] = 0
            stop_update_count[improvement_small] += 1 
            self.update_param_best_value(is_better)
       
            
        if torch.any(stop_update_count >= self.early_stop):
            num_early_stop = torch.sum(stop_update_count >= self.early_stop).item()  
        else:
            num_early_stop = 0

        update = (num_early_stop != len(self._index))
        return  update, (stop_update_count, batched_eff_iter, batched_best_chi_mu)     
  
    def optimizing(self, data, model, sigma): 
        """
        Calculate the residual for this fitting iteration.

        Parameters
        ----------
        data
            The galaxy image data in this batch.
        model
            The galaxy model data in this batch.
        sigma
            The galaxy sigma data in this batch.

        Returns
        -------
        total_residual
            The total residual in this fitting iteration, acts as float.
        chi_mu 
            The residual (acts as chi_mu) for each galaxy.
        """          
        mask, mask_num = self.revise_mask()
        chi = torch.pow((data - model)*mask, 2) / torch.pow(sigma, 2)
        if torch.isnan(model).any():
            nan_mask = torch.isnan(model).any(dim=2).any(dim=1)
            nan_indices = torch.where(nan_mask)
            chi[nan_indices] = 0   
        chi_mu = (torch.sum(chi, dim=[1, 2]) 
                   / (mask_num - len(self.dataset.Profiles.variableParam)))      
        total_residual = torch.sum(chi)
        return total_residual, chi_mu
  
    def fit(self):
        """
        The fitting process.

        After setting the dataset mode in `fitting`, we load it into the 
        data loader.

        Then we make the grid for model generation.

        The galaxies are loaded in batch using the data loader, and in 
        each batch we initialize the fitting metrics, fixed parameter
        values and optim_list.

        When each batch finishes fitting, their img_block will be saved 
        directly. When the total fitting process is finished, the fitted 
        parameter values will be saved in the end.
        """          
        self.dataset.mode = "fitting"
        train_dataset = Data.DataLoader(dataset=self.dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=False)
        self.make_grid()
        
        for (self._index, 
             self.data_cube, 
             self._psf, 
             _param_final_value,
             _consant_value) in train_dataset:
            
            fitting_metrics = self.initialize_fitting_metrics()
            self.load_constant(self.dataset.Profiles.fixedParam, 
                               _consant_value)
            optim_list = self.make_optim_list(
                self.dataset.Profiles.variableParam,
                _param_final_value
                )
            optimizerrr = self.optimizer(optim_list)
            aa = []
            for iter in tqdm(np.arange(self.iteration)):
                self.refresh_variable(self.dataset.Profiles.variableParam, 
                                      optim_list)
                conv_model = self.modeling(
                    self.dataset.Profiles.conv_profile_dict
                    )
                no_conv_model = self.modeling(
                    self.dataset.Profiles.no_conv_profile_dict
                    )
                model_tot = (self.add_convolution(conv_model.sum(axis=0)) 
                             + no_conv_model.sum(axis=0))

                loss, chi_mu = self.optimizing(self.data_cube[:, 0], 
                                                model_tot, 
                                                self.data_cube[:, 1])
                loss.backward()
                optimizerrr.step()
                optimizerrr.zero_grad()
                update, fitting_metrics = self.detect_update(fitting_metrics,
                                                        chi_mu, 
                                                        iter
                                                    ) 
                aa.append(loss.detach().cpu().numpy())
                if not update:
                    break 

            self.fitting_metrics_append(fitting_metrics[1], fitting_metrics[2]) 
            self.append_param_best_value()
            
            if isinstance(self.dataset.img_block_path, str):
                self.save_image()
        
        if isinstance(self.dataset.result_path, str):
            self.save_param()    

        if self.dataset.img_block_path is not None:
            print("The image block is saved in {} successfully!"
                  .format(self.dataset.img_block_path))  
        else:
            print("The image block is not saved becuase"
                  " img_block_path is not set.")     
                
        if self.dataset.result_path is not None:
            print("The parameter fitting result is saved in {} successfully!"
                  .format(self.dataset.result_path))  
        else:
            print("The parameter fitting result becuase"
                  " result_path is not set.")   
      


  



