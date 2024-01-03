from galmoss.profile.ellip_profile import Ellip
import random
from collections.abc import Sequence as Seq
from typing import Union, Tuple, List, Dict
import numpy as np
import torch
from functools import wraps
import torch.utils.data as Data
from astropy.io import fits
import os
import autogalaxy as ag
import autogalaxy.plot as aplt
import autoarray as aa
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def detect_mode(func):
    
    @wraps(func)
    def wrapper(
        cls,
        *args,
        **kwargs
    ) -> np.ndarray:
       
        data = func(cls, *args, **kwargs)
          
        if cls.mode == "wr_uc": # bsp_uc cm_uc wr_uc
          return data
        
        else:
          return torch.from_numpy(data).to(device)
        
      
    return wrapper
  
class Imaging():
  def __init__(
      self, 
      galaxy_index: List[str],
      image_path: List[str],
      psf_path: List[str] = None,
      sigma_path: List[str] = None,
      seg_path: List[str] = None,
      image_hdu: int = 0, 
      psf_hdu: int = 0, 
      sigma_hdu: int = 0, 
      seg_hdu: int = 0
    ):
    self.flex = {}
    self.fix = {}
    self.profiless = []
    self.galaxy_index = galaxy_index
    # add warning
    self.image_path = image_path
    self.image_hdu = image_hdu

    self.psf_path = psf_path
    self.psf_hdu = psf_hdu
    self.sigma_path = sigma_path
    self.sigma_hdu = sigma_hdu
    self.seg_path = seg_path
    self.seg_hdu = seg_hdu
    
    self.use_seg = True if seg_path is not None else False
    self.conv_psf = True if psf_path is not None else False
    

  @detect_mode
  def read_data(self, path, fits_name, hdu):
    HDU = fits.open(os.path.join(path, fits_name + ".fits"))

    return HDU[hdu].data.astype('float32')
  
  @property
  def data_size(self):
    aa = random.randrange(len(self.galaxy_index))
    print(self.image_path, self.galaxy_index, aa)
    HDU = fits.open(os.path.join(self.image_path,
                                 str(self.galaxy_index[aa]) + ".fits"))
    # hdu 可能有好多层 默认选最后一层
    data = HDU[self.image_hdu].data.astype('float32')

    return np.shape(data)

  def cal_radius(self, scale_data, center, item):
    """
    This function is for calculating the reconstruction radius in auto-galaxy package.
    
    center_x:

    """
    if center is None:
      center_x = scale_data.shape[1]
      center_y = scale_data.shape[0]
    else:
      center_x = center[0].value_u[item]
      center_y = center[1].value_u[item]
      
    index = np.where(scale_data == 1)
    radius = np.sqrt(np.power(index[0]-int(center_y), 2) + np.power(index[1]-int(center_x), 2))
    radius = max(radius)
    return radius   


  def perform_fit_with_galaxy(self, imaging, galaxy, radius):
    """
    This function is for make a instance for calculating the regulization.
    imaging, galaxy: instance
    radius: floatshaoe
    """
    mask_2d = ag.Mask2D.circular(
        shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=radius
    )

    imaging = imaging.apply_mask(mask=mask_2d)
    plane = ag.Plane(galaxies=[galaxy])
    return ag.FitImaging(dataset=imaging, plane=plane)

class Profiles(): 
  def __init__(self, **kwargs):
    self.flex = {}
    self.fix = {}
    for name, val in kwargs.items():
      flexx, fixx = val.param(name=name)
      unique_flex = set(self.flex.values()) | set(flexx.values())
      unique_flx = set(self.fix.values()) | set(fixx.values())
      
      self.flex = {index: value for index, value in enumerate(unique_flex, start=1)}      
      self.fix = {index: value for index, value in enumerate(unique_flx, start=1)}
      setattr(self, name, val)
   

  @property
  def profile_dict(self) -> Dict:
      return {
          key: value
          for key, value in self.__dict__.items()
          if isinstance(value, Ellip)
      }
  
  
class Data_Box(Imaging):
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
      seg_path: List[str] = None,
      image_hdu: int = 0, 
      psf_hdu: int = 0, 
      sigma_hdu: int = 0, 
      seg_hdu: int = 0,
      seg_index = None,
    ):
    super().__init__(galaxy_index=galaxy_index, image_path=image_path, psf_path=psf_path, 
                     sigma_path=sigma_path, seg_path=seg_path, image_hdu=image_hdu, psf_hdu=psf_hdu,
                     sigma_hdu=sigma_hdu, seg_hdu=seg_hdu)
    self.Profiles: Profiles = None
    self.seg_index = seg_index
    self.mode = "fitting"
    
    self.data_matrix_index = ["data", "sigma"]

  
  def define_profiles(self, profiles: Profiles):
    
    self.Profiles = profiles
    print(self.Profiles, "aaa")
    
    
  # def bootstrap_fix(self, if_fix):
  #   assert (self.mode == "bs_uc"), "boot_fix can only use under boot-strap mode!"
  #   self.if_fix_when_bootstrap = if_fix

  def param_matrix_from_(self, param_dict, mode, item):
    fix_p = torch.empty((len(param_dict))).to(device)
    
    for ams, param in enumerate(param_dict.values()):
      fix_p[ams] = param.value(mode)[item]

    return fix_p  
        
  def fit_fix_p(self, item):
    fix_p = torch.empty((len(self.Profiles.fix))).to(device)
    
    for ams, param in enumerate(self.Profiles.fix.values()):
      fix_p[ams] = param.proj_Ini_value[item]

    return fix_p  
  

  def fit_flex_p(self, item): #fit_flex_p
    flex_p = torch.empty((len(self.Profiles.flex))).to(device)
    for ams, param in enumerate(self.Profiles.flex.values()): 
      print(np.shape(param.proj_Ini_value()))
      flex_p[ams] = param.proj_Ini_value()[item]

    return flex_p  
      
 
  def bsp_flex_p(self, item):
    flex_p = torch.empty((len(self.Profiles.bsp_flex))).to(device)

    for ams, param in enumerate(self.Profiles.bsp_flex.values()):
        flex_p[ams] = param.proj_fitted_value()[item]
    return flex_p  
 
  def bsp_fix_p(self, item):
    fix_p = torch.empty((len(self.Profiles.bsp_fix))).to(device)

    for ams, param in enumerate(self.Profiles.bsp_fix.values()):
      if param.if_fit == True:
        fix_p[ams] = param.proj_fitted_value()[item]
      else:
        fix_p[ams] = param.proj_Ini_value()[:, item]
    return fix_p 

  def cm_flex_p(self, item):
    flex_p = torch.empty((len(self.Profiles.flex))).to(device)
    for ams, param in enumerate(self.Profiles.flex.values()): 
      flex_p[ams] = param.proj_fitted_value()[item]

    return flex_p  
    

  def sigma_pretreatment(self, data):
    data[data == -99] = 0
    data[data==0] =1   
    return data 
    
  def __getitem__(self, item):
    
    fits_data = self.read_data(self.image_path, self.galaxy_index[item], self.image_hdu)
    # torch.sum(psfim_data) != 0:

    # print(np.shape(fits_data), "shaoe")
    sigma_data = self.sigma_pretreatment(self.read_data(self.sigma_path, self.galaxy_index[item], self.sigma_hdu))

    data_matrix = torch.cat([fits_data.unsqueeze(0), sigma_data.unsqueeze(0)], dim=0)  
    if self.conv_psf:
      psf_data = self.read_data(self.psf_path, self.galaxy_index[item], self.psf_hdu)
    else:
      psf_data = torch.zeros((1)).to(device)
      # data_matrix = torch.cat([data_matrix, psf_data.unsqueeze(0)], dim=0)
      # self.data_matrix_index.append("psf")
    
    if self.use_seg:   
      segim_data = self.read_data(self.seg_path, self.galaxy_index[item], self.seg_hdu)

      if self.seg_index is not None:
        segim_data[segim_data != int(self.seg_index[item])] = 0
        segim_data[segim_data == int(self.seg_index[item])] = 1
      data_matrix = torch.cat([data_matrix, segim_data.unsqueeze(0)], dim=0)
      self.data_matrix_index.append("seg")

    if self.mode == "fitting":
      return self.galaxy_index[item], data_matrix, psf_data, self.param_matrix_from_(self.Profiles.flex, "proj_Ini_value", item), self.fit_fix_p(item)
    elif self.mode == "bstrap":
      return self.galaxy_index[item], data_matrix, psf_data, self.bsp_flex_p(item), self.bsp_fix_p(item)
    elif self.mode == "covar_mat":
      return self.galaxy_index[item], data_matrix, psf_data, self.cm_flex_p(item), self.fit_fix_p(item)
    else:
      assert "Unavailable"
      
    
    
    # elif self.mode == "wr_uc":
    #   reconstruct_fits_data = self.reconstruct_galaxy(item, fits_data, sigma_data, psf_data, segim_data, center=self.reconstruct_c)
    #   return self.galaxy_index[item], reconstruct_fits_data, segim_data, mskim_data, psf_data, sigma_data,  self.bsp_flex_p(item), self.fit_fix_p(item)
    
    
    
    # elif self.mode == "lm_uc":
    #   psf_data = self.read_data(self.psf_path, self.galaxy_index[item], self.psf_hdu).to(device) if self.psf_path != None else torch.zeros((np.shape(fits_data))).to(device)

    #   sigma_data = self.read_data(self.sigma_path, self.galaxy_index[item], self.sigma_hdu).to(device) if self.sigma_path != None else torch.zeros((np.shape(fits_data))).to(device)

    #   if self.seg_path is not None:
    #     segim_data0 = self.read_data(self.seg_path, self.galaxy_index[item], self.seg_hdu).to(device)
    #       # 有些segim读进来包含其他数字，转为0|1矩阵 turn to 0|1 matrix
    #     # segim_data = torch.where((segim_data != 1) & (segim_data != 0), 1, segim_data)
    #     segim_data = segim_data0.clone()
          
    #     if self.seg_index is not None:
    #       index = np.where(self.galaxy_index == self.galaxy_index[item])
    #       segim_data[segim_data != int(self.seg_index[index])] = 0
    #       segim_data[segim_data == int(self.seg_index[index])] = 1
    #     if torch.sum(segim_data) == 0: 
    #       print("segim_data11", self.galaxy_index[item], torch.sum(segim_data0))  
                  
    #   else:
    #     segim_data = torch.ones((np.shape(fits_data))).to(device)

    #   mskim_data = self.read_data(self.mask_path, self.galaxy_index[item], self.mask_hdu).to(device)  if self.mask_path != None else torch.zeros((np.shape(fits_data))).to(device)
      
      
    #   return self.galaxy_index[item], segim_data, mskim_data, psf_data, sigma_data,  self.fit_flex_p(item), self.fit_flex_p(item)
    
    # elif self.mode == "bs_uc":
    #   fits_data = self.read_data(self.image_path, self.galaxy_index[item], self.image_hdu).to(device)
    #   psf_data = self.read_data(self.psf_path, self.galaxy_index[item], self.psf_hdu).to(device) if self.psf_path != None else torch.zeros((np.shape(fits_data))).to(device)

    #   sigma_data = self.read_data(self.sigma_path, self.galaxy_index[item], self.sigma_hdu).to(device) if self.sigma_path != None else torch.zeros((np.shape(fits_data))).to(device)

    #   if self.seg_path is not None:
    #     segim_data0 = self.read_data(self.seg_path, self.galaxy_index[item], self.seg_hdu).to(device)
    #       # 有些segim读进来包含其他数字，转为0|1矩阵 turn to 0|1 matrix
    #     # segim_data = torch.where((segim_data != 1) & (segim_data != 0), 1, segim_data)
    #     segim_data = segim_data0.clone()
          
    #     if self.seg_index is not None:
    #       index = np.where(self.galaxy_index == self.galaxy_index[item])
    #       segim_data[segim_data != int(self.seg_index[index])] = 0
    #       segim_data[segim_data == int(self.seg_index[index])] = 1
    #     if torch.sum(segim_data) == 0: 
    #       print("segim_data11", self.galaxy_index[item], torch.sum(segim_data0))  
                  
    #   else:
    #     segim_data = torch.ones((np.shape(fits_data))).to(device)

    #   mskim_data = self.read_data(self.mask_path, self.galaxy_index[item], self.mask_hdu).to(device)  if self.mask_path != None else torch.zeros((np.shape(fits_data))).to(device)
      
      
    #   return self.galaxy_index[item], fits_data, segim_data, mskim_data, psf_data, sigma_data,  self.bsp_flex_p(item), self.bsp_fix_p(item)
        
  def __len__(self):
    return len(self.galaxy_index)    
  
  
  
    