from ellip import Ellip, Light_Profile
import random
from parameters import Centre_x, Centre_y, Inclination, Axis_ratio, Effective_radius, Intensity, Sersic_index

from sersic import Sersic
from collections.abc import Sequence as Seq
from typing import Union, Tuple, List, Dict
import numpy as np
import torch
from functools import wraps
import torch.utils.data as Data
from astropy.io import fits
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Imaging():
  def __init__(
      self, 
      galaxy_index: List[str],
      image_path: List[str],
      mask_path: List[str],
      psf_path: List[str],
      sigma_path: List[str],
      seg_path: List[str],
      image_hdu: int = 0, 
      mask_hdu: int = 0, 
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
    self.mask_path = mask_path
    self.mask_hdu = mask_hdu
    self.psf_path = psf_path
    self.psf_hdu = psf_hdu
    self.sigma_path = sigma_path
    self.sigma_hdu = sigma_hdu
    self.seg_path = seg_path
    self.seg_hdu = seg_hdu
    


  def read_data(self, path, fits_name, hdu):
    HDU = fits.open(os.path.join(path, fits_name + ".fits"))
    # hdu 可能有好多层 默认选最后一层
    data = HDU[hdu].data.astype('float32')
    
    return torch.from_numpy(data)
  
  @property
  def data_size(self):
    aa = random.randrange(len(self.galaxy_index))
    print(len(self.image_path))
    HDU = fits.open(os.path.join(self.image_path,
                                 self.galaxy_index[aa] + ".fits"))
    # hdu 可能有好多层 默认选最后一层
    data = HDU[self.image_hdu].data.astype('float32')

    return np.shape(data)
    

class Profiles(): 
  def __init__(self, **kwargs):
    self.flex = {}
    self.fix = {}
    self.test = {}
    for name, val in kwargs.items():
      flexx, fixx, test = val.param(name=name)
      print("type", flexx, type(flexx.values))
      self.flex.update(flexx)
      self.fix.update(fixx)
      self.test.update(test)
      setattr(self, name, val)
      
  # @property
  # def flex(self):
    
    
  # @property
  # def fix(self):
    

  @property
  def profile_dict(self) -> Dict:
      return {
          key: value
          for key, value in self.__dict__.items()
          if isinstance(value, Light_Profile)
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
      mask_path: List[str],
      psf_path: List[str],
      sigma_path: List[str],
      seg_path: List[str],
      image_hdu: int = 0, 
      mask_hdu: int = 0, 
      psf_hdu: int = 0, 
      sigma_hdu: int = 0, 
      seg_hdu: int = 0
    ):
    super().__init__(galaxy_index=galaxy_index, image_path=image_path, mask_path=mask_path, psf_path=psf_path, 
                     sigma_path=sigma_path, seg_path=seg_path, image_hdu=image_hdu, mask_hdu=mask_hdu, psf_hdu=psf_hdu,
                     sigma_hdu=sigma_hdu, seg_hdu=seg_hdu)
    self.Profiles: Profiles = None
      


  def from_profiles(self, profiles: Profiles):
    self.Profiles = profiles
    
  
  def __fix_parameters(self, item):
    fixx_param = torch.empty((len(self.Profiles.fix)))
    if len(self.fix) > 0:
      ams=0
      for param in self.fix.values():
        fixx_param[ams] = param.value()[:, item]
        ams=+1
    else:
      fixx_param=[torch.nan]
    return fixx_param
  
     
  def __optim_parameters(self, item):
    optim_param = torch.empty((len(self.Profiles.flex))).to(device)
    ams=0
    # add warning

    for param in self.Profiles.flex.values():
      optim_param[ams] = param.value()[:, item]
      ams+=1

    return optim_param  

        
  def __getitem__(self, item):

    # scale_bias = self.scale_bias[:, item] 
    fits_data = self.read_data(self.image_path, self.galaxy_index[item], self.image_hdu).to(device)
    psf_data = self.read_data(self.psf_path, self.galaxy_index[item], self.psf_hdu).to(device)
    sigma_data = self.read_data(self.sigma_path, self.galaxy_index[item], self.sigma_hdu).to(device)

    segim_data = self.read_data(self.seg_path, self.galaxy_index[item], self.seg_hdu).to(device)
    # 有些segim读进来包含其他数字，转为0|1矩阵 turn to 0|1 matrix
    segim_data = torch.where((segim_data != 1) & (segim_data != 0), 1, segim_data)

      
    mskim_data = self.read_data(self.mask_path, self.galaxy_index[item], self.mask_hdu).to(device)

    
    return self.galaxy_index[item], fits_data, segim_data, mskim_data, psf_data, sigma_data,  self.__optim_parameters(item), self.__fix_parameters(item)
  def __len__(self):
    return len(self.galaxy_index)    
  
  
  
# if __name__ == "__main__":
#   data_shape = [10, 10]
#   x = torch.Tensor(list(range(1, (data_shape[1])+1)))
#   y = torch.flip(torch.Tensor(list(range(1, (data_shape[0])+1))), [0])
#   xy = torch.meshgrid(y, x)
  
#   center_x = Centre_x(parameters=[5])
#   center_y = Centre_y(parameters=[5])
#   incli = Inclination(parameters=100)
#   axi = Axis_ratio(parameters=0.2)
#   effr = Effective_radius(parameters=5)
#   index = Sersic_index(parameters=3)
#   inten = Intensity(parameters=16)
  
#   test = Sersic(centre_x= center_x, centre_y= center_y, inclination=incli, axis_ratio= axi, effective_radius=effr,
#                 sersic_index=index, intensity=inten)

#   aaa = Data_Box(galaxy_index=["1-76"],
#                  image_path="/data/public/ChenM/MIFIT/MANGA/data/test/fitim",
#                  mask_path="/data/public/ChenM/MIFIT/MANGA/data/test/mskim",
#                  psf_path="/data/public/ChenM/MIFIT/MANGA/data/test/psfim",
#                  sigma_path="/data/public/ChenM/MIFIT/MANGA/data/test/sigma/",
#                  seg_path="/data/public/ChenM/MIFIT/MANGA/data/test/segim")
  
#   aaa.from_profiles(Profiles(serssic = test))
#   b = Data.DataLoader(dataset=aaa, batch_size=1, shuffle=True)
#   for galaxy_index,fits_data, segim_data, mskim_data, psfim_data, sigma_data, data, scale_bias in b:
#     print(data)
#     print("-"*20)
#     # for bbb in aaa.Profiles:
#     #   print(bbb)
#     #   print(aaa.serssic)
#     #   print(aaa.bbb)


  


  
    
  
  
    