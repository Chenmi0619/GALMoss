from tqdm import tqdm
import torch_optimizer as optim
import os
import torch.utils.data as Data
from astropy import convolution as conv
import torch.nn.functional as F
from functorch import jacrev
import numpy as np
import torch
import time
from typing import Union, Tuple

from astropy.io import fits

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Bootstrap:
  def __init__(
    self): 
    """
    Class for uncertainty calculation method Bootstrap.
    """ 
    pass
           
  def prepair_data(self, if_fix):
    """
    this function is for changing the flex and fix data dic. If if_fix == True, this means the profile center will be fixed
    """
    if if_fix == True:
      self.dataset.Profiles.bsp_flex = self.dataset.Profiles.flex.copy()
      self.dataset.Profiles.bsp_fix = self.dataset.Profiles.fix.copy()
      for name, param in self.dataset.Profiles.flex.items():
        if isinstance(param, Centre_x) or isinstance(param, Centre_y):
          param.if_fit == False
          next_index = len(self.dataset.Profiles.bsp_fix) + 1
          self.dataset.Profiles.bsp_fix[next_index] = self.dataset.Profiles.bsp_flex.pop(name)

  def resample_segim(self, segim_data):
    """
    this function is bootstrap the data value of galaxy image data. The segim of straped data pixel will keep "1" if it is "1" before, the pixel without strap will be change to "0". The pixel being sample more than one, the segim will change to the number being straped.
    
    segim_data:
      shape: (batch_size, shape[0], shape[1])
      
    new_seg_data:
      shape: (batch_size, shape[0], shape[1])
    """
    
    random_integers = torch.randint(0, np.shape(segim_data)[1] * np.shape(segim_data)[2] - 1,  (np.shape(segim_data)[1] * np.shape(segim_data)[2], ))
    
    seg_data_flat = segim_data.reshape(len(self.galaxy_train_index), -1)
    new_seg_data = torch.zeros(np.shape(seg_data_flat)).to(device)
    
    unique_arr, counts = torch.unique(random_integers, return_counts=True)

    new_seg_data[:, unique_arr] = seg_data_flat[:, unique_arr] * counts.to(device)
    new_seg_data = new_seg_data.reshape(len(self.galaxy_train_index), np.shape(segim_data)[1], np.shape(segim_data)[2])     
    return new_seg_data
  
  

  # def bootstrap_outpram(self, inside_loop=True):
  #   if inside_loop == True:
  #     ams = 0
  #     for param in self.dataset.Profiles.flex.values():
  #       # if if_uncertainty == True:
  #       #   param.uncertainty_store(fi_total[:, ams]) # 10, 1
  #       param.bootstrap_looping_value_store(param.bestValue)
  #       ams = ams + 1
  #       param.grad_clean()    
  #   else:
  #     ams = 0
  #     for param in self.dataset.Profiles.flex.values():
  #       # if if_uncertainty == True:
  #       #   param.uncertainty_store(fi_total[:, ams]) # 10, 1

  #       param.bootstrap_store(param._bootstrap_store)
  #       param._bootstrap_store = torch.Tensor([]).to(device)
  #       ams = ams + 1
  #       param.grad_clean()   
          
  def bootstrap(self, n_resample: int = 500, iteration: int = 100, center_fix: bool = True):
    """
    Calculate the fitting uncertainty for each parameters.

    Parameters
    ----------
    n_resample
        The time of the resampling process. The suggest value is 100.
    iteration
        The maximum fitting iteration, the fitting process will be shut down when reaching this number.
    center_fix
        To decide whether changing the type of paramter (Center_x & Center_y) from free to fix or not, because it is hard to fit when losing the center pixel during the bootstrap resampling process. The suggest value is True.
        
    """
    self.dataset.mode = "bstrap"
    self.prepair_data(center_fix)
    bsp_dataset = Data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
       
    for self.galaxy_train_index,fits_data, segim_data, mskim_data, psfim_data, self.sigma_data, flex_p, fix_p in bsp_dataset:
      chi_total = torch.zeros(len(self.galaxy_train_index), iteration).to(device)

      self.reshaped_Model = torch.zeros(len(self.galaxy_train_index), iteration, int(torch.sum(segim_data, dim = [1, 2]).max().item())).to(device)

      for j in tqdm(np.arange(n_resample)): 
        
        optim_list = self.make_optim_list(self.dataset.Profiles.bsp_flex, flex_p)
        self.load_constant(self.dataset.Profiles.bsp_fix, fix_p)
        optimizerrr = self.optimizer(optim_list, eps=1e-7) 

        new_seg_data = self.resample_segim(segim_data)
        _loss_not_update = torch.zeros((len(self.galaxy_train_index))).to(device)

        for iter in tqdm(np.arange(iteration)):
          self.step_param(self.dataset.Profiles.bsp_flex, optim_list)
          
          model = self.modeling(self._model_cube)
          toal_model = model.sum(axis=0)
          model = self.conv(toal_model, psfim_data, len(self.galaxy_train_index)) if self.dataset.if_psf == True else toal_model

          chi = (torch.pow((fits_data - model).to(device), 2) * new_seg_data * mskim_data )/ torch.pow(self.sigma_data, 2)
          
          if torch.isnan(model).any():
            indices = torch.argwhere(np.isnan(model))[:, 0]   
            chi[indices] = torch.zeros((self.dataset.data_size[0], self.dataset.data_size[1])).to(device)
            chi_miu = torch.sum(chi, dim=[1, 2]) / (torch.sum(new_seg_data * mskim_data, dim=[1, 2]) - len(self.dataset.Profiles.bsp_flex))
            loss = torch.sm(chi_miu)
            chi_miu[indices] = torch.nan
          else:
            chi_miu = torch.sum(chi, dim=[1, 2]) / (torch.sum(new_seg_data * mskim_data, dim=[1, 2]) - len(self.dataset.Profiles.bsp_flex))
            loss = torch.sum(chi_miu)
          
          
          loss.backward()
          optimizerrr.step()      
          optimizerrr.zero_grad()    
          
          _loss_not_update, if_update = self.detect_update(_loss_not_update, chi_miu, iter) 
          if not if_update:
            break 

        chi_total[:, j] = self._batch_chi_miu
        loss.detach()
        self.resample_param_store(self.dataset.Profiles.bsp_flex)
        self.cal_bspError(model, segim_data, j)
        # self.save_bspError(self, self.galaxy_train_index, model, iter)
      # 不同bs的数据保存
      self.resample_param_store(self.dataset.Profiles.bsp_flex, inside_loop=False)
      self.resample_chi_store(chi_total)   
      self.save_bspError()
    self.resample_uc(self.dataset.Profiles.bsp_flex)

  def cal_bspError(self, model, scale, iter):
    scale = scale.reshape(len(self.galaxy_train_index), 1, -1)
    model1 = model
    model = model.view(len(self.galaxy_train_index), 1, -1)
    if iter == 0:
      self.model_index = []    
      for i in np.arange(len(self.galaxy_train_index)): #i~2
        length =int(torch.sum(scale, dim = [1, 2])[i].item())
        index = torch.nonzero(scale[i] == 1)[:, 1]
        self.model_index.append(index)
        sub_model = model[i, 0]
        # print(np.shape(sub_model))
        # print(sub_model[40:50, 40:50])
        self.reshaped_Model[i, iter,:length] = sub_model[index]

    else:
      # sub_model = model[i, 0]
      # self.reshaped_Model[i, iter,:length] = sub_model[index]
      for i in np.arange(len(self.galaxy_train_index)): #i~2
        length =int(torch.sum(scale, dim = [1, 2])[i].item())
        index = self.model_index[i]
        sub_model = model[i, 0]
        # print(np.shape(sub_model))
        # print(sub_model[40:50, 40:50])
        self.reshaped_Model[i, iter,:length] = sub_model[index]
        
      # for i in np.arange(len(self.galaxy_train_index)): #i~2
      #   self.reshaped_Model[i, iter,:self.length] = model[index]      



class CovarianceMatrix:
  def __init__(
    self
    ):  
    
      """
      Class for uncertainty calculation method CovarianceMatrix.
      """
      pass
        
  def covar_mat(self, bs: int =200):
    """
    Calculate the fitting uncertainty for each parameters.

    Parameters
    ----------
    bs
        the batch-size for calculating covariance matrix. Cause of the large memory storage needed for calculation, we suggest to put this value below V:
        0.7 * [ 8 * N_p * N_pixel + N_pixel**2 * 4] < M
        M is the total memory, N_p is the number of free parameters, N_pixel is the maximum pixel number being involved in fitting.
        
    """
        
    self.dataset.mode = "covar_mat"
    cm_dataset = Data.DataLoader(dataset=self.dataset, batch_size=bs, shuffle=False)
    uc_total = torch.empty((0,)).to(device)
    
    
    for self._index, self._data_cube, self._psf, _param_final_value, _consant_value in cm_dataset:
      self.load_constant(self.dataset.Profiles.fix, _consant_value)
      
      self._model_cube = self.make_model_cube(len(self._index))
      with torch.no_grad():
        J = torch.transpose(jacrev(self.for_jaccobian, chunk_size=10)(_param_final_value), 0, 2)
        torch.cuda.empty_cache()
        

      reshaped_J, reshaped_W = self._reconstruct(J)
      biggest_bs = self._cal_biggest_bs()
      
      if biggest_bs > len(self._index) or biggest_bs < 0:
        batch_size = len(self._index)
      else:
        batch_size = biggest_bs

      J_batches = torch.split(reshaped_J, int(batch_size), dim=0)
      sigma_batches = torch.split(reshaped_W, int(batch_size), dim=0)

      

      for i in np.arange(len(J_batches)):

        weight_eye = torch.eye(int(torch.sum(self._seg, dim = [1, 2]).max().item())).to(device).repeat(len(J_batches[i]), 1, 1)
        jTj = torch.bmm(torch.bmm(J_batches[i], weight_eye * sigma_batches[i].float()), J_batches[i].transpose(1, 2))        
        torch.cuda.empty_cache()   
        
        LU, pivots = torch.linalg.lu_factor(jTj.detach())
        del jTj
        del weight_eye
        torch.cuda.empty_cache()
        
        jTj_inv = torch.linalg.lu_solve(LU, pivots, torch.eye(len(self.dataset.Profiles.flex)).to(device).repeat(len(J_batches[i]), 1, 1)) 

        uc = torch.pow(torch.diagonal(jTj_inv, dim1=-2, dim2=-1), 0.5)
        uc_total = torch.concat((uc_total, uc), dim=0)
        

    for ams, param in enumerate(self.dataset.Profiles.flex.values()):
      param.uncertainty_store(uc_total[:, ams]) # 10, 1
      ams+=1
      
  def _reconstruct(self, J:torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    """
    To extract the effective value from weight matrix and Jacobian matrix based on the segmentation matrix.

    Parameters
    ----------
    J
        The Jacobian matrix
    """
    self._weight = (1 / (torch.pow(self._data_cube[:, 1] , 2))).view(len(self._index), 1, -1)      
    self._seg = self._data_cube[:, 2].reshape(len(self._index), 1, -1)
    self._max_length = int(torch.sum(self._seg, dim = [1, 2]).max().item())
    reshaped_J = torch.zeros(len(self._index), len(self.dataset.Profiles.flex), self._max_length, dtype=self._weight.dtype).to(device)
    reshaped_W = torch.zeros(len(self._index), 1, self._max_length, dtype=self._weight.dtype).to(device)

    value_index = []
    for i in np.arange(len(self._index)): #i~2
      length =int(torch.sum(self._seg, dim = [1, 2])[i].item())
      galaxy_weight = self._weight[i, 0]
      index = torch.nonzero(self._seg[i] == 1)[:, 1]
      value_index.append(index)

      reshaped_W[i, 0, : length] = galaxy_weight[index]

      
      for k in np.arange(len(self.dataset.Profiles.flex)): #k~7
        reshaped_J[i, k, : length] = J[k, i][index]
    return reshaped_J, reshaped_W
          
  def _cal_biggest_bs(self) -> int:
    """
    To calculate the biggest batch_size available under the given memory storage.
    """
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device=device)
    cached_memory = torch.cuda.memory_reserved(device=device)
    free_memory = total_memory - allocated_memory - cached_memory

    occupied_memory = (4 * self._max_length **2) + 2 * 4 * len(self.dataset.Profiles.flex) * self._max_length
    biggest_bs = int(0.7*free_memory / occupied_memory) 
    return biggest_bs
          
  def for_jaccobian(self, param: torch.Tensor) -> torch.Tensor:
    """
   This function **would not** be used to calculate model value, only to be load in jacrevm for calculating the Jacobian matrix. 
    
    Parameters
    ----------
    param
        The result of parameters after fitting process.
    """
    for ams, param_class in enumerate(self.dataset.Profiles.flex.values()):
      data = param[:, ams]
      # print("data", data)
      param_class.update(data)

    model = self.modeling(self._model_cube, mode="updating_value") # 1， batch, 128, 128
    conv_model_sky = self.make_total_sky(model.sum(axis=0))
    model = torch.sum(conv_model_sky, dim=0).view((model.shape[2] * model.shape[3]))


    return model           
    
    
            
class Uncertainty(Bootstrap, CovarianceMatrix):
  def __init__(
    self, 
  ):  
    self.resample_chi = torch.empty((0,)).to(device)


  def resample_chi_store(self, chi_bs):
    self.resample_chi = torch.concat((self.resample_chi, chi_bs), dim=0) 
        
  def resample_param_store(self, P_dict, inside_loop=True):
    """
    This function is for storing the parameters fitting value inside resampling process(no matter which kind).
    if inside_loop equals True, which means the matrix is not complete in total, but only in iteration dimension or fits number dimension.
    P_dict: the dict of flex parameters
    param._loop_store_bs: the incomplete storing matrix.
      shape: (1, iteration ) or (1, galaxy_n)
    param.resampling_Value: the complete resampling value matrix.
      shape: iteration, galaxy_n
    """
    if inside_loop == True:
      for param in P_dict.values():
        param._loop_store_bs = torch.concat((param._loop_store_bs, param.bestValue.reshape(1, -1)), dim=0)
        param.grad_clean()    
    else:
      for param in P_dict.values():
        param.resampling_Value = torch.concat((param._loop_store_total, param._loop_store_bs), dim=1) # 500, 1000
        param._loop_store_bs = torch.Tensor([]).to(device)
        param.grad_clean()
        
  def resample_uc(self, P_dict):
    for param in P_dict.values():
      param.uncertainty = torch.std(param.resampling_Value, dim=0)


