import numpy as np
import torch
import pandas as pd
from typing import Union, Tuple

from galmoss.Parameter import decorater  


class ParamsRepo:
     
    def __init__(self):   
        """
        This repository stores various types of values with specific
        naming conventions:

        1. 'total' vs 'batched': 
            'batched' variables change with each galaxy batch during
            fitting, whereas 'total' variables accumulate values across
            batches, completing only after the entire fitting process.

        2. Domain types - 'mapped', 'normalized', or unspecified: 
            Variables are initially in the 'original input domain'. 
            'mapped' variables are in the 'profile equation domain' 
            relevant for PA, scale-related variables (e.g., eff_r), and 
            mag. 'normalized' variables, often used in optimizers, are 
            scaled to (-1~1) when range constraints apply.

        3. Value states - 'initial', 'updating', 'best': 
            'initial' variables are the starting value in the parameter
            object; 'updating' variables change during fitting; 'best' 
            variables are the final fitted value.

        4. Storage formats - 'cpu', 'numpy', or unspecified: 
            Variables with 'cpu' are tensors on CPU; with 'numpy' are 
            numpy.array on CPU; unspecified ones are tensors on 
            user-defined devices, typically GPU (cuda) by default.

        Value names and comments indicate their nature, e.g., 
        'self.total_mapped_initial_val_cpu' signifies initial total values 
        in the original input domain, stored as tensors on CPU. A value 
        dictionary "self.value_dict" allows quick indexing. Changes here 
        require corresponding updates in the `value` function's ValueError 
        information.
        """        
        if self.log:
            self.total_mapped_inital_val_cpu = torch.pow(10, self._parameters)
        else:
            self.total_mapped_inital_val_cpu = self._parameters
        self.value_dict = {
            "updating_model": "batched_broadcasted_updating_val", 
            "best_model": "batched_broadcasted_mapped_best_val",
            "load_best_in_optim_list": "total_normalized_best_val_cpu",
            "load_best_in_jacobian": "total_mapped_best_val_cpu",
            "load_initial_in_jacobian": "total_mapped_ini_value", 
            "load_initial_in_optim_list": "total_normalized_ini_value", 
            "initial_model": "total_broadcasted_mapped_inital_val",
        }    
            
    def value(self, mode: str = "updating_model") -> torch.tensor:
        """
        Return the value guided by the 'mode' argument.
        
        Parameters
        ----------
        mode
            The string representing which value to choose from self.value_dict.
        """              
        if mode not in self.value_dict:
            raise ValueError(
                "Wrong mode text, should be one of: updating_model, best_model, "
                "load_best_in_optim_list, load_best_in_jacobian, "
                "load_initial_in_optim_list, or initial_model"
            )
        
        return getattr(self, self.value_dict[mode])
 
    @property
    def batched_broadcasted_updating_val(self):
        """
        Return the updating value in each sub-fitting batch, mapped to 
        the profile equation domain 
        (e.g., mag' = mag - M0, eff_r' = eff_r / pix_scale), and 
        broadcasted for calculations. This is typically used to generate
        the model image during the fitting process.

        This attribute can also be accessed via 
        param.value("updating_model").
        """   
        if self.batched_updating_value is not None:
            if self.dim == "1d":
                return self.batched_updating_value.unsqueeze(1) 
        else:
            return None        

    @property
    def total_broadcasted_mapped_inital_val(self, device: str = "cpu"): 
        """
        Return the total initial value, mapped to the profile equation
        domain (e.g., mag' = mag - M0, eff_r' = eff_r / pix_scale), and
        broadcasted for calculations. The data is loaded on the CPU by
        default, but can be changed to other devices using the 'device' 
        argument. This is typically used to generate the initial model 
        image for a quick review.
        
        This attribute can also be accessed by 
        param.value("initial_model").
        """         
        if self.total_mapped_inital_val_cpu is not None:
            return self.total_mapped_inital_val_cpu.unsqueeze(1).to(device)
        else:
            return None        

    @property
    def total_initial_value_numpy(self): 
        """
        Return the total initial value in the original input domain,
        with shape (param_length, dim_length). This is typically used to
        save the fitted parameter values for fixed parameters, as they
        only have an initial value.
        """          
        if self.total_initial_value_cpu_tensor is not None:
            return self.total_initial_value_cpu_tensor.numpy()
        else:
            return None        
                 
    @property
    @decorater.to_normalization
    def total_normalized_ini_value(self):
        """
        Return the total initial value in the normalized domain. This is 
        typically for loading into the optim_list that will later be
        loaded into optimizer in the fitting process. 

        This attribute can also be accessed by 
        param.value("load_initial_in_optim_list").
        """              
        return self._parameters

  
    @property
    @decorater.inv_from_log
    def total_mapped_ini_value(self):
        """
        Return the total initial value in the normalized domain. This is 
        typically for loading into the optim_list that will later be
        loaded into optimizer in the fitting process. 

        This attribute can also be accessed by 
        param.value("load_initial_in_optim_list").
        """              
        return self._parameters
    
      
    @property
    def total_mapped_best_val_cpu(self): 
        """
        Return the total best value, mapped to the profile equation
        domain (e.g., mag' = mag - M0, eff_r' = eff_r / pix_scale), and 
        broadcasted for calculations. This is typically used to generate
        the model image during the fitting uncertainty estimation 
        process using co-variance matrix.
        
        This attribute can also be accessed by 
        param.value("load_best_in_jacobian").
        """       
        if hasattr(self, "total_mapped_best_value"):
            if self.dim == "1d":
                return self.total_mapped_best_value.detach().cpu()
        else:
            return None

    @property 
    @decorater.inv_from_mapping
    def total_best_value(self):
        """
        Return the total best value in the original input domain,
        with shape (param_length, dim_length), in numpy.array. 
        """       
        if hasattr(self, "total_mapped_best_value"):
            return self.total_mapped_best_value.detach()
        else:
            return None

    @property 
    @decorater.inv_from_mapping
    def total_bsp_uncertainty(self):
        """
        Return the total uncertainty in the original input domain,
        with shape (param_length, dim_length), in torch.Tensor. 
        """       
        if hasattr(self, "batched_mapped_best_value"):
            return self.batched_mapped_best_value.detach()
        else:
            return None
        
    @property 
    def total_best_value_numpy(self):
        """
        Return the total best value in the original input domain,
        with shape (param_length, dim_length), in numpy.array. This is 
        typically used to save the fitted values for variable parameters.
        """       
        if hasattr(self, "total_best_value"):
            return self.total_best_value.cpu().numpy() 
        else:
            return None
        
    @property
    def batched_broadcasted_mapped_best_val(self):
        """
        Return the best value in each sub-fitting batch, mapped to 
        the profile equation domain 
        (e.g., mag' = mag - M0, eff_r' = eff_r / pix_scale), and 
        broadcasted for calculations. This is typically used to generate
        the model image in the result image saving process.

        This attribute can also be accessed via 
        param.value("best_model").
        """           
        if self.batched_mapped_best_value is not None:
            if self.dim == "1d":
                return self.batched_mapped_best_value.unsqueeze(1)
        else:
            return None

    @property
    @decorater.to_normalization
    @decorater.to_log
    def total_normalized_best_val_cpu(self):
        """
        Return the total best value in the normalized domain. This is 
        typically for loading into the optim_list that will later be
        loaded into optimizer in the bootstrapping process. 

        This attribute can also be accessed by 
        param.value("load_best_in_optim_list").
        """     
        if hasattr(self, "total_mapped_best_value"):
            return self.total_mapped_best_value.detach().cpu()
        else:
            return None
                     
    @property
    def total_uncertainty_numpy(self):
        """
        Return the total uncertainty value in the original input domain,
        with shape (param_length, dim_length). This is typically used to
        save the fitting uncertainty for variable parameters.
        """  
        if hasattr(self, "uncertainty"):
            return self.uncertainty.detach().cpu().numpy() 
        else:
            return None

  
class Parameters(ParamsRepo):

    def __init__(
        self, 
        parameters: Union[np.ndarray, torch.Tensor, float, pd.Series], 
        value_range:  Union[Tuple[float, float], None] = (None, None),
        step_length: float = 0.015,
        M0: float = None,
        pix_scale: Union[float, None] = None,
        angle: bool = False,
        fit: bool = True,
        log: bool = False,  
        dim: str = "1d"
    ):  
        """
        In GALMOSS, each parameter is represented as an individual object 
        instantiated from the Parameters class. The Parameters class is a 
        fundamental class in GALMOSS. Other classes such as Dataset, 
        Profiles, and fitting classes, can access the attributes of the 
        Parameters class.

        Parameters
        ----------
        parameters : Union[np.ndarray, torch.Tensor, float, pd.Series]
            The original input value of the parameter.
        value_range : Union[Tuple[float, float], None], optional
            Constrains the fitting range.
        step_length : float, optional
            The optimization step length in gradient descent (learning rate 
            in PyTorch's optimizer), by default 0.015.
        M0 : float, optional
            The magnitude zero point for magnitude parameters.
        pix_scale
            The conversion factor between arcseconds and pixels, applicable 
            only to scale-related parameters like eff_r.
        angle
            Indicates if the parameter is angle-related.
        fit
            Indicates if the parameter is variable or fixed during fitting.
        log
            Determines if the parameter should be fitted in log space.
        dim 
            Specifies if the parameter is two-dimensional ('1d' or '2d'). 
            This argument will only be changed in user-defined profile that 
            is very special! 
            
        Attributes
        ----------
        normalization_relationship
            Calculates the normalization relationship based on the provided 
            value_range.

        dim2_length
            In built-in profiles, this attribute is 1. For 2-D parameters, 
            this attribute changes to 2.

        param_length
            The length of the parameter, which is equal to the number of 
            galaxies.            
        """                
        if sum([M0 is not None, 
                pix_scale is not None, angle is not False]) > 1:
            raise ValueError("A parameter can only be one of Magnitude, PA or "
                             "scale_related parameter.")
        if dim in ["1d", "2d"]:
            self.dim = dim
        else:
            raise ValueError(f"The dim '{dim}' is not valid.")

        self.M0 = M0
        if pix_scale is not None:
            self.pix_scale = np.array(pix_scale)
        else:
            self.pix_scale = pix_scale
        self.angle = angle
        self.fit = fit
        self.log = log
        self.step_length = step_length
        self._parameters, self.value_range = self.mapping(parameters, 
                                                          value_range,
                                                          M0, self.pix_scale, 
                                                          angle)
        self.dim2_length = self._parameters.shape[1]
        self.param_length = self._parameters.shape[0]

        if value_range != (None, None):
            self.normalization_relationship = (
                (self.value_range[1] - self.value_range[0]) / 2,
                1 - self.value_range[1] / ((self.value_range[1] - self.value_range[0]) / 2)
            )
            condition = (
                torch.max(self._parameters) <= self.value_range[1] and
                torch.min(self._parameters) >= self.value_range[0]
            )
            assert condition, ("Data is out of the specified range. Please check"
                                "the input values.")
        else:
            self.normalization_relationship = (1, 0)

        super().__init__()
   
    
    @decorater.redim
    @decorater.to_mapping_log_with_value_range
    def mapping(self, 
                parameter: torch.tensor, 
                value_range: Tuple[float, float], 
                *args, 
                **kwargs):
        """
        Returns the mapped parameter and value range. If log=True, this 
        function returns the log of the mapped values.

        Parameters
        ----------
        parameter
            The parameter value in the original input domain.
        value_range
            Constrains the fitting range, with values in the original 
            input domain.
        *args & kwargs
            Additional arguments, can only be M0, pix_scale, or angle. 
            M0 is the magnitude zero point for magnitude parameters. 
            pix_scale is the conversion factor between arcseconds and 
            pixels, applicable only to scale-related parameters like 
            eff_r. angle indicates if the parameter is angle-related.
        """
        return parameter, value_range

    @decorater.inv_from_normalization 
    @decorater.inv_from_log          
    def update_updating_val(self, value: torch.tensor):
        """
        Returns the mapped parameter and value range. If log=True, this
        function returns the log of the mapped values.

        Parameters
        ----------
        parameter
            The parameter value in the original input domain.
        value_range
            Constrains the fitting range, with values in the original
            input domain.
        *args&**kwargs
            Additional arguments, can only be M0, pix_scale, or angle.
            M0 is the magnitude zero point for magnitude parameters. 
            pix_scale is the conversion factor between arcseconds and 
            pixels, applicable only to scale-related parameters like 
            eff_r. angle indicates if the parameter is angle-related.
        """           
        self.batched_updating_value = value

    def update_best_val(self, index: Union[torch.tensor, bool] = None):
        """
        Update the best value during the optimization process. Not every 
        iteration in the optimization process results in a better outcome
        with a lower chi-square. Therefore, updates are made only in 
        iterations that successfully reduce the chi-square, guided by 
        the given index.

        Parameters
        ----------
        index
            Specifies the position in the best value matrix to be
            refreshed. If None, refreshes the entire matrix.
        """               
        if index is None:
            self.batched_mapped_best_value = self.batched_updating_value
        else:
            self.batched_mapped_best_value[index] = self.batched_updating_value[index]
   
    def best_value_append(self, batched_mapped_best_value: torch.tensor):
        """
        Append the best fitted values for each batch of galaxies after
        the completion of each batch fitting process.

        Parameters
        ----------
        batched_mapped_best_value
            The best fitted value for each batch of galaxies.
        """         
        if hasattr(self, 'total_mapped_best_value'):
            self.total_mapped_best_value = torch.concat(
                (self.total_mapped_best_value, batched_mapped_best_value), dim=0)                  
        else:
            self.total_mapped_best_value = batched_mapped_best_value


    @decorater.inv_from_mapping_uncertainty
    def inv_from_mapping_uc(self, data):
        """
        Return the total uncertainty in the original input domain,
        with shape (param_length, dim_length), in numpy.array. 
        """       
        return data
 
 
    # @decorater.inv_from_normalization_and_log_uncertainty
    # def inv_from_normalization_and_log_uc(self, uc, data):
    #     """
    #     Return the total uncertainty in the original input domain,
    #     with shape (param_length, dim_length), in numpy.array. 
    #     """       
    #     return uc     
    
      
    def calculate_uncertainty(self, proj_uc):
        """
        This function is for calculating the final uncertainty of parameters. 
        Inside function "cm_uncertainty", we get the projected uncertianty 
        `self.uncertainty`of parameters, so we need the error propagation 
        equation to expand.
        
        Parameters
        ----------
        proj_uc
            The projected uncertianty of parameters           
        """    
        if self.log:
            real_uc = (
                torch.exp(
                    (proj_uc - self.normalization_relationship[1]) 
                    * self.normalization_relationship[0]
                ) 
                * self.normalization_relationship[0] 
                * proj_uc
            )
        else:
            real_uc = self.normalization_relationship[0] * proj_uc
            
        if hasattr(self, 'uncertainty_gpu'):
            self.uncertainty = torch.concat(
                (self.uncertainty_gpu, real_uc)
                , dim=0
                )                   
        else:
            self.uncertainty = real_uc
            
    def __repr__(self) -> str:
        return self.__class__.__name__    