import numpy as np
import torch
from astropy.io import fits
from typing import List

from galmoss.profile.light_profile import LightProfile


class Profiles(): 
  def __init__(self, **kwargs):
    """
    This class stores the galaxy profiles that will be fitted.

    Parameters
    ----------
    **kwargs: 
        The galaxy profiles, e.g., profile1 = sersic
        
    Attributes
    ----------
    lightProfile_dict
        The dictionary stores all the defined galaxy profiles.
    no_conv_profile_dict
        The dictionary stores the profiles should not be convolved with PSF image.
    conv_profile_dict
        The dictionary stores the profiles should be convolved with PSF image.
    variableParam
        The dictionary stores all the parameters should be fitted.
    fixedParam
        The dictionary stores all the parameters should not be fitted.

    Note:
        The repeat parameters will appear only once in the Param dictionaries.
        For example, if two profiles share the galaxy center, the cen_x and cen_y will
        only appear once in dictionary they belong to.
    """     
    self.lightProfile_dict = {}
    self.no_conv_profile_dict = {}
    self.conv_profile_dict = {} 
    self.variableParam = {}
    self.fixedParam = {}
    unique_variable_param= set()
    unique_fixed_param = set()    
    for name, val in kwargs.items():
        if isinstance(val, LightProfile):
            self.lightProfile_dict[name] = val
            if getattr(val, 'psf', False):
                self.conv_profile_dict[name] = val
            else:
                self.no_conv_profile_dict[name] = val
            val.manage_params(name=name)
            unique_variable_param = unique_variable_param | set(val.variable_dic.values())
            unique_fixed_param = unique_fixed_param | set(val.fixed_dic.values())
        else:
            raise ValueError("The object be load in Profile class should be light profile!")    

    for param in unique_variable_param:
        count = sum(key.startswith(param.param_name) for key in self.variableParam.keys())
        new_key = param.param_name if count == 0 else f"{param.param_name}_{count}"
        self.variableParam[new_key] = param
    for param in unique_fixed_param:
        count = sum(key.startswith(param.param_name) for key in self.fixedParam.keys())
        new_key = param.param_name if count == 0 else f"{param.param_name}_{count}"
        self.fixedParam[new_key] = param
    setattr(self, name, val)


class DataSet():
    """
    Load the images and Parameters in parallel

    Parameters
    ----------
    func
        A function which returns the parameters need to be load in the optimism.

    Returns
    -------
        A function that returns the scale-parameters need to be load in the optimism.
    """
    def __init__(
        self, 
        galaxy_index: List[str],
        image_path: List[str],
        sigma_path: List[str],
        psf_path: List[str] = None,
        mask_path: List[str] = None,
        image_hdu: int = 0,  
        sigma_hdu: int = 0, 
        psf_hdu: int = 0,
        mask_hdu: int = 0,
        mask_index: List[str] = None,
        mask_mode = "bg",
        result_path = None,
        result_type = "FITS",
        img_block_path = None,
        data_type: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        This class stores the important attributes.

        Parameters
        ----------
        galaxy_index: 
            The list of the galaxy names.
        image/sigma/psf/mask_path
            The list of the galaxy/sigma/psf/mask image path.
        image/sigma/psf/mask_hdu
            The hdu number of the galaxy/sigma/psf/mask images.
        mask_index
            The list of the mask index, representing which region belongs
            to the correponding galaxy.
        mask_mode
            `bg`: The mask will only delete the region belongs to other 
            celestial bodies.
            `exclusive`: The mask will only remain the region belongs to 
            the corresponding galaxy.
        result_path
            The saving path of fitted value of parameters.
        result_type
            The data type of fitted value of parameters, should be FITS, 
            CSV or HDF5.
        img_block_path
            The saving path of fitted value of galaxy img_block, which includes
            galaxy image, model image, residual image and sub-component images.
        data_type
            The data type of matrix during calculations
        device
            Calculate the fitting process on CPU or GPU.

        Attributes
        ----------
        mode
            To change the dataset mode into fitting, bootstrap or covar_mat. 
        """     
        self.galaxy_index = [galaxy_index] if isinstance(galaxy_index, str) else galaxy_index 
        self.image_path = [image_path] if isinstance(image_path, str) else image_path 
        self.image_hdu = image_hdu

        self.psf_path = [psf_path] if isinstance(psf_path, str) else psf_path 
        self.psf_hdu = psf_hdu
        self.sigma_path = [sigma_path] if isinstance(sigma_path, str) else sigma_path 
        self.sigma_hdu = sigma_hdu
        self.mask_path = [mask_path] if isinstance(mask_path, str) else mask_path 
        self.mask_hdu = mask_hdu
        
        if mask_mode not in ["bg", "exclusive"]:
            raise ValueError(
                "Wrong mask_mode text, should be one of: bg, exclusive!"
            )   
        self.mask_mode = mask_mode

        self.mask_index = [mask_index] if isinstance(mask_index, int) else mask_index
        if ((mask_path is not None) and (self.mask_index is None)):
            self.mask_index = np.ones((len(self.galaxy_index)))
        
        if not (isinstance(result_path, str) or result_path is None):
            raise ValueError("The type of result_path should be str !")    
        self.result_path = result_path
        
        if result_type not in ["FITS", "CSV", "HDF5"]:
            raise ValueError("Result_type should be one of FITS, CSV or HDF5!")
        self.result_type = result_type

        if not (isinstance(img_block_path, str) or img_block_path is None):
            raise ValueError("The type of img_block_path should be str !")    
        self.img_block_path = img_block_path    
        self.data_type = data_type  
        self.device = device

        self.mode = "fitting"
        
    def read_data(self, 
                  path: str, 
                  hdu: int):
        """
        To read the data from fits file.
        """
        HDU = fits.open(path)
        return torch.from_numpy(HDU[hdu].data.astype(np.float64)).to(self.device)
  
    @property
    def data_size(self):
        """
        To get the data size from the first galaxy's image.
        """        
        HDU = fits.open(self.image_path[0])
        data = HDU[self.image_hdu].data.astype(np.float64)
        return np.shape(data)

    @classmethod
    def define_profiles(cls, *args, **kwargs):  
        """
        To define the profile class and set as an attribute in DataSet class.
        """              
        cls.Profiles = Profiles(**kwargs)

    def param_matrix_from_(self, 
                           param_dict: dict, 
                           mode: str, 
                           item: int):
        """
        To concat the param value matrix following the param_dict, mode and item.
        
        Parameters
        ----------
        param_dict
            The dictionary stores the keys (name of parameters) and values 
            (the parameter objects). These parameters need to be cancated 
            to a matrix that will be loaded into optimizer.
        mode
            The values of parameters will be called follow the mode in the 
            parameter objects, e.g., updating_model.
        item
            The index of values, which will be given automaticly when the 
            dataset is loaded in the data loader.    
        
        Returns
        -------
        total_matrix
            The finial matrix, being transformed to the given data type.            
        """   
        total_matrix = torch.empty((0,))
        for param in param_dict.values():
            if param.pix_scale is not None and hasattr(param, 'pix_scale_device')== False:
                param.pix_scale_device = torch.tensor(param.pix_scale).to(self.device).reshape(-1, 1)            
            if param.value(mode).shape[0] < len(self.galaxy_index):
                raise ValueError("The size of parameter doesn't match the"
                "number of galaxy, expect batch of galaxies with number of"
                "{} have a size of parameter of {}, but get {}".format(
                    len(self.galaxy_index), len(self.galaxy_index), 
                    param.value(mode).shape[1])) 
            total_matrix = torch.concat((total_matrix, param.value(mode)[item]), dim=0)          
        return total_matrix.to(self.data_type)

    def sigma_pretreatment(self, sigma_datadata: torch.tensor):
        """
        To reset the illegal sigma values.
        """         
        max_sigma = torch.max(sigma_datadata)
        sigma_datadata[sigma_datadata <= 0] = max_sigma  
        return sigma_datadata 
    
    def __getitem__(self, item: int):
        """
        Output the data once the DataSet being loaded by data loader.
        """ 
        self.data_cube_idx = ["data", "sigma"]
        fits_data = self.read_data(self.image_path[item], self.image_hdu).to(self.data_type)
        sigma_data = self.sigma_pretreatment(self.read_data(self.sigma_path[item], self.sigma_hdu)).to(self.data_type)

        data_matrix = torch.cat([fits_data.unsqueeze(0), sigma_data.unsqueeze(0)], dim=0)  
        if self.psf_path is not None:
            psf_data = self.read_data(self.psf_path[item], self.psf_hdu).to(self.data_type)
        else:
            psf_data = torch.zeros((1)).to(self.device)

        if self.mask_path is not None:   
            mask_data = self.read_data(self.mask_path[item], self.mask_hdu).to(self.data_type)
            # if self.mask_index is not None:
            if self.mask_mode == "bg":
                mask_data[(mask_data != self.mask_index[item]) & (mask_data != 0)] = 999
                mask_data[mask_data != 999] = 1
                mask_data[mask_data == 999] = 0
            else:
                mask_data[mask_data != int(self.mask_index[item])] = 0
                mask_data[mask_data == int(self.mask_index[item])] = 1
            data_matrix = torch.cat([data_matrix, mask_data.unsqueeze(0)], dim=0)
            self.data_cube_idx.append("seg")
        
        if self.mode == "fitting":
            return (
                self.galaxy_index[item], 
                data_matrix, 
                psf_data, 
                self.param_matrix_from_(
                    self.Profiles.variableParam, 
                    "load_initial_in_optim_list", 
                    item).to(self.device), 
                self.param_matrix_from_(
                    self.Profiles.fixedParam, 
                    "load_initial_in_optim_list", 
                    item).to(self.device)
            )
        elif self.mode == "bootstrap":         
            return (
                self.galaxy_index[item], 
                data_matrix, 
                psf_data, 
                self.param_matrix_from_(
                    self.Profiles.bsp_variableParam, 
                    "load_best_in_optim_list", 
                    item).to(self.device), 
                self.param_matrix_from_(
                    self.Profiles.bsp_fixedParam, 
                    "load_initial_in_optim_list", 
                    item).to(self.device)
                )
        elif self.mode == "covar_mat":
            return (
                self.galaxy_index[item], 
                data_matrix, 
                self.param_matrix_from_(
                    self.Profiles.variableParam, 
                    "load_best_in_jacobian", 
                    item).to(self.device), 
                self.param_matrix_from_(
                    self.Profiles.fixedParam, 
                    "load_initial_in_optim_list", 
                    item).to(self.device)
                )
        else:
            assert "Unavailable"
        
    def __len__(self):
        return len(self.galaxy_index)    
  
  
  
    