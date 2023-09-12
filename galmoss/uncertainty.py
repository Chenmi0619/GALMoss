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
from galmoss_1.Parameter.parameters import Centre_x, Centre_y
from astropy.io import fits

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Bootstrap:
  def __init__(
    self):  
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
          
  def bootstrap(self, iteration=100, fits_n=500, boot_fix = True, aspError=False):
    self.dataset.mode = "bsp_uc"
    
    self.prepair_data(self.dataset.if_fix_when_bootstrap)
    bsp_dataset = Data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
       
    for self.galaxy_train_index,fits_data, segim_data, mskim_data, psfim_data, self.sigma_data, flex_p, fix_p in bsp_dataset:
      chi_total = torch.zeros(len(self.galaxy_train_index), iteration).to(device)

      self.reshaped_Model = torch.zeros(len(self.galaxy_train_index), iteration, int(torch.sum(segim_data, dim = [1, 2]).max().item())).to(device)

      for j in tqdm(np.arange(iteration)): 
        
        optim_list = self.make_optim_list(self.dataset.Profiles.bsp_flex, flex_p)
        self.set_constant(self.dataset.Profiles.bsp_fix, fix_p)
        optimizerrr = self.optimizer(optim_list, eps=1e-7) 

        new_seg_data = self.resample_segim(segim_data)
        _loss_not_update = torch.zeros((len(self.galaxy_train_index))).to(device)

        for iter in tqdm(np.arange(fits_n)):
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

  def save_bspError(self):
    Empty_bspError = torch.zeros(self.dataset.data_size[0]*self.dataset.data_size[1]).to(device)
    checkkk = torch.zeros(100, self.dataset.data_size[0]*self.dataset.data_size[1]).to(device)

    # for ams, index in enumerate(galaxy_index):
    #   empty_aspError[value_index] = value
    #   aspError = empty_aspError.view(self.dataset.data_size[0], self.dataset.data_size[1])
    for ams, index in enumerate(self.galaxy_train_index):
      empty_bspAveError = Empty_bspError.clone()
      empty_bspStdError = Empty_bspError.clone()

      error_index = self.model_index[ams]
      model = self.reshaped_Model[ams]
      
      ave = torch.mean(model, dim=(0))
      std = torch.std(model, dim=(0))

      for i in np.arange(100):
        checkkk[i][error_index] = model[i][0: empty_bspAveError[error_index].shape[0]]
      np.save("/data/public/ChenM/MIFIT/MANGA/data/g/final_paper_data/111/model.npy", checkkk.detach().cpu().numpy())              
      empty_bspAveError[error_index] = ave[0: empty_bspAveError[error_index].shape[0]]
      empty_bspStdError[error_index] = std[0: empty_bspAveError[error_index].shape[0]]

      bspAveError = empty_bspAveError.view(self.dataset.data_size[0], self.dataset.data_size[1])
      bspStdError = empty_bspStdError.view(self.dataset.data_size[0], self.dataset.data_size[1])
      
      bspAve_hdu = fits.ImageHDU(bspAveError.detach().cpu().numpy())
      bspStd_hdu = fits.ImageHDU(bspStdError.detach().cpu().numpy())
      bspAve_hdu.header['KEYWORD'] = 'bspAveError'
      bspStd_hdu.header['KEYWORD'] = 'bspStdError'

      hdul = fits.open(os.path.join(self.result_path, index + '.fits'))
      hdul.append(bspAve_hdu)
      hdul.append(bspStd_hdu)
      
      hdul.writeto(os.path.join(self.result_path, index + '.fits'), overwrite=True)



class CovarianceMatrix:
  def __init__(
    self):  
    pass

  def save_aspError(self, galaxy_index, value, value_index): 
    Empty_aspError = torch.zeros(self.dataset.data_size[0]*self.dataset.data_size[1]).to(device)
    
    for ams, g_index in enumerate(galaxy_index):
      index = value_index[ams]
      sub_value = value[ams]
      empty_aspError = Empty_aspError.clone()
      print("aaa", empty_aspError[index].shape[0], empty_aspError[index].shape, np.shape(sub_value))
      empty_aspError[index] = sub_value[0: empty_aspError[index].shape[0]]
      aspError = empty_aspError.view(self.dataset.data_size[0], self.dataset.data_size[1])
      aspError_hdu = fits.ImageHDU(aspError.detach().cpu().numpy())
      aspError_hdu.header['KEYWORD'] = 'aspError'
      hdul = fits.open(os.path.join(self.result_path, g_index + '.fits'))
      hdul.append(aspError_hdu)
      
      hdul.writeto(os.path.join(self.result_path, g_index + '.fits'), overwrite=True)
     

  def cm_uncertainty(self, bs, aspError=False):
    self.dataset.mode = "cm_uc"
    cm_dataset = Data.DataLoader(dataset=self.dataset, batch_size=bs, shuffle=False)
    fi_total = torch.Tensor([]).to(device)
    
    
    for self.uc_galaxy_train_index, self.uc_data_matrix, flex_p, fix_p in cm_dataset:
      scale = self.uc_segim_data.reshape(len(self.uc_galaxy_train_index), 1, -1)

      if self.dataset.if_psf:
        self.uc_weight_input_psf =  self.data_matrix[:, self.dataset.data_matrix_index("psf")].unsqueeze(1)
        
      self._uc_model_cube = torch.zeros((len(self.dataset.Profiles.profile_dict), 
                                len(self.uc_galaxy_train_index), 
                                self.dataset.data_size[0], 
                                self.dataset.data_size[1])).to(device)

      self.set_constant(self.dataset.Profiles.fix, fix_p)
      with torch.no_grad():
        ft_jacobian = jacrev(self.for_jaccobian1, chunk_size=10)(flex_p) # 1, 16384, 7, i [16384, 7]
        torch.cuda.empty_cache()
      J = torch.transpose(ft_jacobian, 0, 2)
      # print('j'*30, J)
      del ft_jacobian


      _weight = (1/(torch.pow(self.data_matrix[:, 1] , 2))).view(len(self.uc_galaxy_train_index), 1, -1)

      scale = self.uc_segim_data.reshape(len(self.uc_galaxy_train_index), 1, -1)

      reshaped_J = torch.zeros(len(self.uc_galaxy_train_index), len(self.dataset.Profiles.flex), int(torch.sum(scale, dim = [1, 2]).max().item()), dtype=J.dtype).to(device)
      reshaped_W = torch.zeros(len(self.uc_galaxy_train_index), 1, int(torch.sum(scale, dim = [1, 2]).max().item()), dtype=_weight.dtype).to(device)

      
      value_index = []
      for i in np.arange(len(self.uc_galaxy_train_index)): #i~2
        length =int(torch.sum(scale, dim = [1, 2])[i].item())
        galaxy_weight = _weight[i, 0]
        index = torch.nonzero(scale[i] == 1)[:, 1]
        value_index.append(index)

        reshaped_W[i, 0, : length] = galaxy_weight[index]

        
        for k in np.arange(len(self.dataset.Profiles.flex)): #k~7
          # a = J[k, i] # [2, 7, 16384
          reshaped_J[i, k, : length] = J[k, i][index]

      
      total_memory = torch.cuda.get_device_properties(device).total_memory
      allocated_memory = torch.cuda.memory_allocated(device=device)
      cached_memory = torch.cuda.memory_reserved(device=device)
      free_memory = total_memory - allocated_memory - cached_memory

      occupied_memory = 4 * np.shape(reshaped_J)[2] + (4 * np.shape(reshaped_J)[2] * np.shape(reshaped_J)[2]) * 2 + 4 * len(self.dataset.Profiles.flex) * np.shape(reshaped_J)[2]
      biggest_bs = int(0.7*free_memory / occupied_memory)  # For 7 x 16384
      if biggest_bs > len(self.uc_galaxy_train_index) or biggest_bs < 0:
        batch_size = len(self.uc_galaxy_train_index)
      else:
        batch_size = biggest_bs

      J_batches = torch.split(reshaped_J, int(batch_size), dim=0)
      sigma_batches = torch.split(reshaped_W, int(batch_size), dim=0)
      split_index_tuple = tuple(self.uc_galaxy_train_index[i:i+batch_size] for i in range(0, len(self.uc_galaxy_train_index), int(batch_size)))
      

      for i in np.arange(len(J_batches)):
        
        name = split_index_tuple[i]

        weight_eye = torch.eye(int(torch.sum(scale, dim = [1, 2]).max().item())).to(device).repeat(len(J_batches[i]), 1, 1)
        jTj = torch.bmm(torch.bmm(J_batches[i], weight_eye * sigma_batches[i].float()), J_batches[i].transpose(1, 2))
        # print(reshaped_J, "reshaped_J")


        torch.cuda.empty_cache()
                
        jTj = jTj.detach()
        LU, pivots = torch.linalg.lu_factor(jTj)

        del jTj
        torch.cuda.empty_cache()
        
        jTj_inv = torch.linalg.lu_solve(LU, pivots, torch.eye(len(self.dataset.Profiles.flex)).to(device).repeat(len(J_batches[i]), 1, 1)) 
        print("aaa", np.shape(1/sigma_batches[i]), np.shape(torch.bmm(torch.bmm(J_batches[i].transpose(1, 2), jTj_inv), J_batches[i])))
        # sigma_y = torch.pow(torch.diagonal(weight_eye/sigma_batches[i]+torch.bmm(torch.bmm(J_batches[i].transpose(1, 2), jTj_inv), J_batches[i]), dim1=-2, dim2=-1), 0.5)
        sigma_y = torch.pow(torch.diagonal(torch.bmm(torch.bmm(J_batches[i].transpose(1, 2), jTj_inv), J_batches[i]), dim1=-2, dim2=-1), 0.5)
        fi = torch.pow(torch.diagonal(jTj_inv, dim1=-2, dim2=-1), 0.5)
        if aspError == True:
          self.save_aspError(name, sigma_y, value_index)
        

        
        # print(np.shape(fit_sigma), "aaa")
        
        del weight_eye
        # self.save_aspError(sigma_y)
        fi_total = torch.concat((fi_total, fi), dim=0)
        print("fi", fi)
        

    for ams, param in enumerate(self.dataset.Profiles.flex.values()):
      param.uncertainty_store(fi_total[:, ams]) # 10, 1
      ams+=1
          

  


# class NoiseResample:
#   def __init__(
#     self):  
#     pass
  
#   def resample(self, fits_data):
#     return torch.noraml(mean=fits_data, std=torch.sqrt(fits_data))
   
            
#   def wr_uncertainty(self, arc2pix, center: Union[Tuple[Centre_x, Centre_y], Tuple[None, None]]=(None, None), iteration=100, fits_n=100):
#     self.dataset.arc2pix = arc2pix
#     self.dataset.reconstruct_c = center
#     self.dataset.mode = "wr_uc"
    

#     wr_dataset = Data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
       
#     for self.galaxy_train_index,reconstruct_fits_data, segim_data, mskim_data, psfim_data, self.sigma_data, flex_p, fix_p in wr_dataset:
#       chi_total = torch.zeros(len(self.galaxy_train_index), iteration).to(device)

#       self._model_cube = torch.zeros((len(self.dataset.Profiles.profile_dict), 
#                                 len(self.galaxy_train_index), 
#                                 self.dataset.data_size[0], 
#                                 self.dataset.data_size[1])).to(device)
#       self.set_constant(self.dataset.Profiles.fix, fix_p)
      
#       for j in tqdm(np.arange(iteration)): 
        
#         optim_list = self.make_optim_list(self.dataset.Profiles.flex, flex_p)
#         optimizerrr = self.optimizer(optim_list, eps=1e-7) 
#         new_fits_data = self.resample(reconstruct_fits_data)
#         _loss_not_update = torch.zeros((len(self.galaxy_train_index))).to(device)

#         for iter in tqdm(np.arange(fits_n)):
#           self.step_param(self.dataset.Profiles.bsp_flex, optim_list)
          
#           model = self.modeling(self._model_cube)
#           toal_model = model.sum(axis=0)
#           model = self.conv(toal_model, psfim_data, len(self.galaxy_train_index)) if self.dataset.if_psf == True else toal_model

#           chi = (torch.pow((new_fits_data - model).to(device), 2) * segim_data * mskim_data )/ torch.pow(self.sigma_data, 2)
          
#           if torch.isnan(model).any():
#             indices = torch.argwhere(np.isnan(model))[:, 0]   
#             chi[indices] = torch.zeros((self.dataset.data_size[0], self.dataset.data_size[1])).to(device)
#             chi_miu = torch.sum(chi, dim=[1, 2]) / (torch.sum(segim_data * mskim_data, dim=[1, 2]) - len(self.dataset.Profiles.flex))
#             loss = torch.sum(chi_miu)
#             chi_miu[indices] = torch.nan
#           else:
#             chi_miu = torch.sum(chi, dim=[1, 2]) / (torch.sum(segim_data * mskim_data, dim=[1, 2]) - len(self.dataset.Profiles.flex))
#             loss = torch.sum(chi_miu)
          
          
#           loss.backward()
#           optimizerrr.step()      
#           optimizerrr.zero_grad()    
          
#           _loss_not_update, if_update = self.detect_update(_loss_not_update, chi_miu, iter) 
#           if not if_update:
#             break 

#         chi_total[:, j] = self._batch_chi_miu
#         loss.detach()
#         self.resample_outpram()
#       # 不同bs的数据保存
#       self.resample_outpram(self.dataset.Profiles.flex, inside_loop=False)
#       self.resample_chi_store(chi_total)   
#     self.resample_uc()
    
    
            
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


