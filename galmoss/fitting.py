from galmoss_1.data import Data_Box
from galmoss_1.uncertainty import Uncertainty
import numpy as np
import torch
import time
from functools import wraps
from tqdm import tqdm
import torch_optimizer as optim
import os
import torch.utils.data as Data
from astropy import convolution as conv
import torch.nn.functional as F
from functorch import jacrev
from astropy.io import fits


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def conv_addition(func):
    """
    To check if the total model needs conv or not

    Parameters
    ----------
    func
        A function which returns the total model 

    Returns
    -------
        A function that returns the conved model 
    """
    @wraps(func)
    def wrapper(
        cls,
        total_model,
        *args,
        **kwargs
    ) -> np.ndarray:
        """

        Parameters
        ----------
        cls : Parameters
            The class that owns the function.

        Returns
        -------
            The conved total model with sky
        """

        if cls.dataset.if_psf == True:
          return func(cls, cls.conv(total_model, cls.psf_data, len(cls.galaxy_train_index)), *args, **kwargs)
        else:
          return func(cls, total_model, *args, **kwargs)
    return wrapper
  
def sky_addition(func):
    """
    To check if the total model after convolution needs to add sky or not

    Parameters
    ----------
    func
        A function which returns the model after convolution psf function

    Returns
    -------
        A function that returns the conved model with sky
    """
    @wraps(func)
    def wrapper(
        cls,
        total_model,
        *args,
        **kwargs
    ) -> np.ndarray:
        """

        Parameters
        ----------
        cls : Parameters
            The class that owns the function.

        Returns
        -------
            The conved total model with sky
        """

        if hasattr(cls.dataset.Profiles, "sky"):
          return func(cls, total_model + cls.dataset.Profiles.sky.image_2d_grid_from_(cls.grid), *args, **kwargs)
        else:
          return func(cls, total_model, *args, **kwargs)

    return wrapper
  
class FittingRepo():
  def __init__(self):
    self.effect_iter = torch.empty((0,))
    self._chi_miu = torch.empty((0,)).to(device)
    
    self.falure_index = torch.empty((0,)).to(device)
  
  def fit_initialRepo(self, bs_len):
    """
    This function is for initial the repository, for storing the new batch's fitting result.
    """
    self._batch_effect_iter = torch.zeros((bs_len))
    self._batch_chi_miu = torch.empty((0,)).to(device)

  def bootstraping_initialize(self, bs_len):
    self._batch_effect_iter = torch.zeros((bs_len))
    self._batch_chi_miu = torch.empty((0,)).to(device)
         
  def store_effect_iter(self, batch_effect_iter):
    """
    used to store the last updating iterations after one batch's traing, concating to final chi_matrix
    sub_iter: (bs)
    """    
    self.effect_iter = torch.concat((self.effect_iter, batch_effect_iter), dim=0) 

  def loss_store(self, chi_miu, index=None):
    """
    iter_chi_miu: (1, bs)
    index: the production after torch.where
    _updating_chi_miu: (1, bs)
    self.loss_repo: (iter, bs)
    """
    if index == None:
      self._batch_chi_miu = chi_miu
    else:
      self._batch_chi_miu[index] = chi_miu[index]



  def bootstrap_chi_store(self, aaa):
    if len(self.bs_chi) == 0:
      self.bs_chi = aaa
    else:
      self.bs_chi = torch.concat((
        self.bs_chi, aaa), dim=0)  
                         
  def store_chi_miu(self, batch_chi_miu):
    """
    Used to store the last updating chi after iterations in one batch,
    concatenating it to the final chi_matrix.

    Parameters
    ----------
    batch_chi_miu : torch.Tensor
        The chi values to be stored for the current batch.

    Returns
    -------
    None
    """
    self._chi_miu = torch.concat((self._chi_miu, batch_chi_miu), dim=0)   

  @property
  def chi_miu(self) -> np.ndarray:
      """
      Used to turns the chi_miu data after fitting into numpy

      Returns
      -------
      np.ndarray
          The chi_miu values as a NumPy array.
      """
      return self._chi_miu.detach().cpu().numpy()

          
class Fitting(FittingRepo, Uncertainty):
  """
  Base class for Parameters

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
      dataset: Data_Box,
      batch_size: int,
      iteration: int,
      optimizer = optim.DiffGrad,
      early_stop = 50,
      robust = 0.01,
      if_uncertainty = False,
      result_path = None,
      save_decoposition = False
      
  ):
    super().__init__()
    self.dataset = dataset
    self.batch_size = batch_size
    self.iteration = iteration
    self.optimizer = optimizer
    
    self.sub_factor = 1
    self.grid = 0
    self.early_stop = early_stop
    self.robust = robust

    self.bs_chi = []
    self.name = []
    self.if_uncertainty = if_uncertainty
    # self.model = []
    if result_path is None:
      self.save_model = False
    else:
      self.save_model = True
      self.result_path = result_path
    
    self.aaa = 0
    self.bbb = 0
    Uncertainty.__init__(self)
  

  def make_optim_list(self, P_dict, P_value): # optimizer_param_list
      optim_list = []
      for ams, param in enumerate(P_dict.values()): 
          param_dict = {}
          param_value = P_value[:, ams]
          param_dict['params'] = param_value.detach().requires_grad_(True)
          param_dict['lr'] = param.step_length
          optim_list.append(param_dict)
      return optim_list
                 
       
  def modeling(self, model_cube, mode="updating_value"):

    model = model_cube.clone()
    for ams, key in enumerate(self.dataset.Profiles.profile_dict):
      profile = self.dataset.Profiles.profile_dict[key]
      time1 = time.time()
      model[ams]= profile.image_2d_grid_from_(self.grid, mode)
      time2 = time.time()
      self.aaa += time2-time1
      self.bbb +=1
    return model
      
  
  def make_grid(self):
      x = torch.linspace(0.5, self.dataset.data_size[1]-0.5, self.dataset.data_size[1]).to(device)
      y = torch.linspace(self.dataset.data_size[0]-0.5, 0.5, self.dataset.data_size[0]).to(device)
      xy = torch.meshgrid(y, x, indexing='ij')      
      self.grid = xy

  
  def step_param(self, P_dict, optim_list):
    """
    This function is for refresh the stepped value inside optimizer to the parameters instance, in sequence of the parameter diction being given.
    
    P_dict:
    the flex parameters diction
    optim_list:
    the flex parameters diction that is produced by P_dict before
    """
    for ams, param in enumerate(P_dict.values()):
      data = optim_list[ams]['params'][0]
      param.update(data)

    
  def set_constant(self, P_dict, P_value):
    """
    This function is for set the fix parameters (constant) before every batch-size of training.
    """
    for ams, param in enumerate(P_dict.values()):
      param.update(P_value[:, ams].detach())
      param.refresh_bv(index=None)        
  
  
  def Refresh_best_param(self, index):
    ams=0
    with torch.no_grad(): 
      for param in self.dataset.Profiles.flex.values():
        param.refresh_bv(index)
        
        ams += 1

      
  def save_param(self):
    """
    This function is for save the bestValue to the value repository inside parameters instances.
    bestValue:
      shape: (batch_size)
    """
    for param in self.dataset.Profiles.flex.values():
      param.value_store(param.bestValue)
      param.grad_clean()
    

  def restore_param(self):
    """
    This function is for make a tensor to be put inside jacobbian function. The value is extract out from the param's best-Value
    param.proj_fitted_value():
      shape: (n_flex_data)
    param_tensor:
      shape: (n_flex_data, n_galaxy)
    """    
    param_tensor = torch.Tensor([]).to(device)

    for param in self.dataset.Profiles.flex.values():
      param_tensor = torch.concat((param_tensor, param.proj_fitted_value().unsqueeze(0)), dim=0)
    return param_tensor
  
  def for_jaccobian1(self, param):
    for ams, param_class in enumerate(self.dataset.Profiles.flex.values()):
      data = param[:, ams]
      # print("data", data)
      param_class.update(data)



    model = self.modeling(self._uc_model_cube, mode="best_value") # 1， batch, 128, 128
    shape = np.shape(model)
    antisky_model = model.sum(axis=0) # bactch, 128, 128
    # antisky_model_input_psf = antisky_model.unsqueeze(0).to(dtype=torch.float)

    # aaa = torch.sum(self.weight_input_psf[0], dim=[1, 2])
    # # print("test22334455", np.shape(antisky_model_input_psf), np.shape(self.weight_input_psf))
    # antisky_model_after_psf = F.conv2d(antisky_model_input_psf,self.uc_weight_input_psf,padding="same", groups=len(self.uc_galaxy_train_index)) #/ aaa
    # antisky_model_after_psf = torch.squeeze(antisky_model_after_psf)
    # aaa = torch.unsqueeze(torch.unsqueeze(aaa, 1), 1)
    # antisky_model_after_psf = antisky_model_after_psf/aaa

    # model = antisky_model_after_psf
    model = torch.sum(antisky_model, dim=0)
    model = model.view((shape[2] * shape[3]))
    
    # b = a[0, 1].view((shape[2] * shape[3]))
    # print(np.shape(a))

    return model 
    
    
  def loss_store(self, chi_miu, index=None):
    """
    iter_chi_miu: (1, bs)
    index: the production after torch.where
    _updating_chi_miu: (1, bs)
    self.loss_repo: (iter, bs)
    """
    if index == None:
      self._batch_chi_miu = chi_miu
    else:
      self._batch_chi_miu[index] = chi_miu[index]
       

  def save_image(self):
    model = self.modeling(model_cube=self._model_cube, mode="best_value")
    component_cube = self._model_cube.clone()
    # toal_model = model.sum(axis=0)
    # conv_model = self.conv(toal_model, self.psfim_data, len(self.galaxy_train_index)) if self.dataset.if_psf == True else toal_model
    # conv_model_sky = conv_model + self.dataset.Profiles.sky.image_2d_grid_from_(self.grid)
    toal_model = self.make_total_sky(model.sum(axis=0))    
    residual = self.data_matrix[:, 0] - toal_model

    if model.shape[0] > 1:
      for ams1, key in enumerate(self.dataset.Profiles.profile_dict):    
        component_cube[ams1] = self.make_total(model[ams1])          
    
    for ams, galaxy in enumerate(self.galaxy_train_index):
      primary_hdu = fits.PrimaryHDU()
      image_hdu = fits.ImageHDU(self.data_matrix[:, 0][ams].detach().cpu().numpy())
      image_hdu.header['KEYWORD'] = 'data'
      
      model_hdu = fits.ImageHDU(toal_model[ams].detach().cpu().numpy())
      model_hdu.header['KEYWORD'] = 'model'
          
      residual_hdu = fits.ImageHDU(residual[ams].detach().cpu().numpy())
      residual_hdu.header['KEYWORD'] = 'residual'
      

      # Create a new HDU list and append the HDUs
      hdulist = fits.HDUList([primary_hdu, image_hdu, model_hdu, residual_hdu])
      component_hdulist = [primary_hdu, image_hdu, model_hdu, residual_hdu]

      if model.shape[0] > 1:
        
        for ams1, key in enumerate(self.dataset.Profiles.profile_dict):
          component_hdu = fits.ImageHDU(component_cube[ams1, ams].detach().cpu().numpy())
          component_hdu.header['KEYWORD'] = key
          component_hdulist.append(component_hdu)       
      # Save the HDU list to a FITS file
      component_hdulist = fits.HDUList(component_hdulist)
      component_hdulist.writeto(os.path.join(self.result_path, galaxy + '.fits'), overwrite=True)

    
  def residual_store(self, aaa):
    aaa = [tensor.detach().cpu().numpy() for tensor in aaa]
    if self.save_residual == True:
      if not os.path.exists(self.save_residual_path):
        os.mkdir(self.save_residual_path)
      path = os.path.join(self.save_residual_path, "residual.npy")
      if os.path.exists(path):
        res = np.load(path, allow_pickle=True)
        res = np.concatenate((res, aaa), axis=0)  
        np.save(path, res)
      else:
        np.save(path, aaa)
    else:
      pass
      



      
  
  def conv(self, model_data, psf_data, batch_size):
    """
    This function is for calculating the convolution of paralleled psf data and model data
    """

    noramlize_factor = torch.sum(psf_data, dim=[1, 2]).unsqueeze(1).unsqueeze(1)
    
    return torch.squeeze(F.conv2d(model_data.unsqueeze(0), psf_data.unsqueeze(1), padding="same", groups=batch_size)) / noramlize_factor 
  
  @sky_addition
  @conv_addition
  def make_total_sky(self, total_model):
    return total_model


  @conv_addition
  def make_total(self, total_model):
    return total_model
        
  def train(self):

    self.dataset.mode = "fitting"
    
    train_dataset = Data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
    self.make_grid()
    for self.galaxy_train_index, self.data_matrix, self.psf_data, flex_data, fix_data in train_dataset:
      
      self.fit_initialRepo(len(self.galaxy_train_index))
    
      self.set_constant(self.dataset.Profiles.fix, fix_data)
                
      self._model_cube = torch.zeros((len(self.dataset.Profiles.profile_dict), 
                                len(self.galaxy_train_index), 
                                self.dataset.data_size[0], 
                                self.dataset.data_size[1])).to(device)
      
      # 初始化
      
      _loss_not_update = torch.zeros((len(self.galaxy_train_index))).to(device)
      optim_list = self.make_optim_list(self.dataset.Profiles.flex, flex_data)
      optimizerrr = self.optimizer(optim_list, eps=1e-7) 
      
      for iter in tqdm(np.arange(self.iteration)):
        self.step_param(self.dataset.Profiles.flex, optim_list)
        model = self.modeling(self._model_cube)
        conv_model_sky = self.make_total_sky(model.sum(axis=0))
        # conv_model_sky = self.conv(model.sum(axis=0), self.psfim_data, len(self.galaxy_train_index)) + self.dataset.Profiles.sky.image_2d_grid_from_(self.grid) if self.dataset.if_psf == True else model.sum(axis=0) + self.dataset.Profiles.sky.image_2d_grid_from_(self.grid)
        # index = input_list.index(target)
        
        if self.dataset.if_psf:
          chi = torch.pow((self.data_matrix[:, 0] - conv_model_sky) * self.data_matrix[:, self.dataset.data_matrix_index.index("seg")], 2) / torch.pow(self.data_matrix[:, 1], 2)  
           
          chi_miu = torch.sum(chi, dim=[1, 2]) / (torch.sum(self.data_matrix[:, self.dataset.data_matrix_index.index("seg")], dim=[1, 2]) - len(self.dataset.Profiles.flex) )
        else:
          chi = torch.pow((self.data_matrix[:, 0] - conv_model_sky), 2) / torch.pow(self.data_matrix[:, 1], 2)   
          chi_miu = torch.sum(chi, dim=[1, 2]) / (self.data_matrix[:, 0].shape[0] * self.data_matrix[:, 0].shape[1] - len(self.dataset.Profiles.flex) )
        
        loss = torch.sum(chi_miu)
        print(loss)
        # if any galaxy model's fitting failed, we put the chi to zero, and mark the galaxy index to self.falure_index
        if torch.isnan(model).any():
          indices = torch.argwhere(torch.isnan(model))[:, 0]   
          chi[indices] = torch.zeros((self.dataset.data_size[0], self.dataset.data_size[1])).to(device)
          self.falure_index = torch.concat((self.falure_index, indices))


        optimizerrr.zero_grad()
        loss.backward()
        optimizerrr.step()

        _loss_not_update, if_update = self.detect_update(_loss_not_update, chi_miu, iter) 
        if not if_update:
          break 

      # 修改合适的名字
      # if self.save_model == True:
      #   self.result_store(model)
      del model, conv_model_sky
      self.store_chi_miu(self._batch_chi_miu) 
      self.store_effect_iter(self._batch_effect_iter)
      loss.detach()
      self.save_param()
      


      self.save_image()
    # self.finish_process()

    
  
  def detect_update(self, n_not_uopdate, chi_miu, iter):
    if iter == 0:
      n_not_uopdate += 1
      self.Refresh_best_param(index=None)
      self.loss_store(chi_miu)
      if_update = True 
      
    else: 
      index = (chi_miu < self._batch_chi_miu) & (self._batch_chi_miu - chi_miu > 0.005)
      self._batch_effect_iter[index] = iter
      self._batch_chi_miu[index] = chi_miu[index]
      n_not_uopdate[index] = 0
      n_not_uopdate[~index] += 1 
      self.Refresh_best_param(index)
      self.loss_store(chi_miu, index)
      
      ams = len(torch.where(n_not_uopdate > self.early_stop)[0]) if len(torch.where(n_not_uopdate > self.early_stop)) != 0 else 0

      if (len(self.galaxy_train_index) - ams) < ( 1 if self.robust*len(self.galaxy_train_index)< 1 else self.robust*len(self.galaxy_train_index)):
        if_update = False
      else:
        if_update = True   
    return  n_not_uopdate, if_update
           
           
  def finish_process(self):
    """
    This function if for detecting the index of galaxy fitting that is undone.
    
    undone_index: the index that its effective iteration number equals total iteration(that means until the last iteration, the fitting still not keep stable) 
    self.undonefit_index: the galaxy name of undone_index
    param.fitting_bestValue: After this iterations of fitting, the best value of each parameters.
    param.undone_fit_value: After this iteration, the temporary best value of the ondone galaxy fitting.
    """
    undone_index = np.where(self.effect_iter == self.iteration -1)
    self.undonefit_index = self.dataset.galaxy_index[undone_index]
    for param in self.dataset.Profiles.flex.values():
      param.fitting_bestValue = param.fit_value  
      param.undone_fit_value = param.fit_value[undone_index]
      
      
  # def ucucuc(self, bs):
  #   self.dataset.mode = "lm_uc"
  #   d = Data.DataLoader(dataset=self.dataset, batch_size=bs, shuffle=False)
  #   fi_total = torch.Tensor([]).to(device)
    
    
  #   for self.uc_galaxy_train_index, self.uc_data_matrix, flex_data, fix_data in d:
  #     scale = self.uc_segim_data.reshape(len(self.uc_galaxy_train_index), 1, -1)
  #     a = torch.sum(scale, dim=[1, 2])

      
  #     self.uc_weight_input_psf = psfim_data.unsqueeze(1)
  #     self.uc_model_cube = torch.zeros((len(self.dataset.Profiles.profile_dict), 
  #                               len(self.uc_galaxy_train_index), 
  #                               self.dataset.data_size[0], 
  #                               self.dataset.data_size[1])).to(device)
  #     optim_list = self.make_optim_list(self.dataset.Profiles.flex, flex_data)
  #     ams=0
  #     self.step_param(self.dataset.profiles.flex, optim_list)
  #     param = self.restore_param() # 7, 1024
  #     # print("paramm", param)
  #     for paramm in self.dataset.Profiles.fix.fit_value():
        
  #       _data = fix_data[:, ams]
  #       # print(param, _data)
  #       ams += 1
  #       paramm.update(_data)
  #       paramm.refresh_bv(index=None)
  #     # a = self.modeling() # 1， batch, 128, 128

  #     initial_memory = torch.cuda.memory_allocated()
  #     with torch.no_grad():
  #       ft_jacobian = jacrev(self.for_jaccobian1, chunk_size=10)(param) # 1, 16384, 7, i [16384, 7]
  #       torch.cuda.empty_cache()

  #     initial_memory = torch.cuda.memory_allocated()
      
      
  #     J = torch.transpose(ft_jacobian, 0, 2)

      
  #     del ft_jacobian

  #     sigma = self.uc_sigma_data 
  #     sigma[sigma==0] =1
  #     _weight = (1/(torch.pow(sigma, 2))).view(len(self.uc_galaxy_train_index), 1, -1)
  #     # _weight[_weight==1] =0

      
      
  #     # row_lengths = torch.count_nonzero(J[:, 0], dim=1)
  #     # print(int(torch.sum(scale, dim = [1, 2]).max().item()))

  #     # a = torch.where(row_lengths != torch.count_nonzero(_weight[:, 0], dim=1))
  #     # print(row_lengths[a], torch.count_nonzero(_weight[:, 0], dim=1)[a], torch.sum(scale[a]))
  #     # print()
  #     scale = self.uc_segim_data.reshape(len(self.uc_galaxy_train_index), 1, -1)

  #     reshaped_J = torch.zeros(len(self.uc_galaxy_train_index), len(self.dataset.Profiles.flex), int(torch.sum(scale, dim = [1, 2]).max().item()), dtype=J.dtype).to(device)
  #     reshaped_W = torch.zeros(len(self.uc_galaxy_train_index), 1, int(torch.sum(scale, dim = [1, 2]).max().item()), dtype=_weight.dtype).to(device)
  #     # reshaped_J = torch.zeros(len(self.uc_galaxy_train_index), len(self.dataset.Profiles.flex), 128*128, dtype=J.dtype).to(device)
  #     # reshaped_W = torch.zeros(len(self.uc_galaxy_train_index), 1, 128*128, dtype=_weight.dtype).to(device)
      
  #     for i in np.arange(len(self.uc_galaxy_train_index)): #i~2
  #       a = torch.sum(scale, dim = [1, 2]).detach().cpu().numpy() 
  #       index = np.where(a == 0)
  #       length =int(torch.sum(scale, dim = [1, 2])[i].item())
  #       # length = 128*128
  #       b = _weight[i, 0]
  #       index = torch.nonzero(scale[i] == 1)[:, 1]
  #       # print(i, length, np.shape(b[index]), index, scale[i, 0].item())

  #       reshaped_W[i, 0, : length] = b[index]
  #       # reshaped_W[i, 0, : length] = b
        
  #       for k in np.arange(len(self.dataset.Profiles.flex)): #k~7
  #         a = J[i, k] # [2, 7, 16384
  #         reshaped_J[i, k, : length] = a[index]
  #         # reshaped_J[i, k, : length] = a
          

          
      
  #     total_memory = torch.cuda.get_device_properties(device).total_memory
  #     allocated_memory = torch.cuda.memory_allocated(device=device)
  #     cached_memory = torch.cuda.memory_reserved(device=device)
  #     free_memory = total_memory - allocated_memory - cached_memory

  #     # Set the batch size based on the available memory
  #     occupied_memory = 4 * np.shape(reshaped_J)[2] + (4 * np.shape(reshaped_J)[2] * np.shape(reshaped_J)[2]) * 2 + 4 * len(self.dataset.Profiles.flex) * np.shape(reshaped_J)[2]
  #     biggest_bs = int(0.7*free_memory / occupied_memory)  # For 7 x 16384



      
      
  #     if biggest_bs > len(self.uc_galaxy_train_index) or biggest_bs < 0:
  #       batch_size = len(self.uc_galaxy_train_index)
  #     else:
  #       batch_size = biggest_bs
  #     # batch_size = 1
  #     J_batches = torch.split(reshaped_J, int(batch_size), dim=0)
  #     sigma_batches = torch.split(reshaped_W, int(batch_size), dim=0)

      
  #     weight_eye_0 = torch.eye(int(torch.sum(scale, dim = [1, 2]).max().item())).to(device).repeat(batch_size, 1, 1)
  #     # weight_eye_0 = torch.eye(128*128).to(device).repeat(batch_size, 1, 1)
  #     initial_memory = torch.cuda.memory_allocated()

  #     for i in np.arange(len(J_batches)):
        
  #       j = J_batches[i]
  #       lennn = len(j)
  #       weight_eye_0 = torch.eye(int(torch.sum(scale, dim = [1, 2]).max().item())).to(device).repeat(lennn, 1, 1)
  #       # weight_eye_0 = torch.eye(128*128).to(device).repeat(batch_size, 1, 1)
  #       w = sigma_batches[i]
  #       weight_eye = weight_eye_0.clone()
  #       # print(np.shape(weight_eye))
  #       weight = weight_eye * w

        

  #       del w, weight_eye
  #       torch.cuda.empty_cache()
  #       j = j.float()

  #       # print("jac_tuple1", j)
  #       jTj = torch.bmm(torch.bmm(j.float(), weight.float()), j.transpose(1, 2))


  #       del weight
  #       torch.cuda.empty_cache()
  #       jTj = jTj.detach()
  #       LU, pivots = torch.linalg.lu_factor(jTj)
  #       jtj_di = torch.diagonal(jTj, dim1=-2, dim2=-1)
  #       # print("k", i)
  #       # if i == 0:
      
  #         # print("jtj_diiiiii", jtj_di, torch.sum(scale, dim = [1, 2])[0: len(jtj_di)], jtj_di/torch.unsqueeze(torch.sum(scale, dim = [1, 2])[0: len(jtj_di)], dim=1))
  #       del jTj, j
  #       t = torch.eye(len(self.dataset.Profiles.flex)).to(device)
        
  #       b = t.repeat(lennn, 1, 1).to(torch.float32)

        
  #       jTj_inv = torch.linalg.lu_solve(LU, pivots, b) 
        
  #       fi = torch.pow(torch.diagonal(jTj_inv, dim1=-2, dim2=-1), 0.5) 
  #       if i == 0:
      
  #         print("fiiii", fi)
  #       fi_total = torch.concat((fi_total, fi), dim=0)
        
  #   ams=0
  #   # print(param)
  #   for paramm in self.dataset.Profiles.flex.values():
  #     # print(np.shape(fi_total[:, ams]), "fi_total[:, ams]")
  #     paramm.uncertainty_store(fi_total[:, ams]) # 10, 1
  #     ams+=1
          

   

  def bootstrap_chi_store(self, aaa):
    if len(self.bs_chi) == 0:
      self.bs_chi = aaa
    else:
      self.bs_chi = torch.concat((
        self.bs_chi, aaa), dim=0)   


# weight_eye = torch.eye(len(_weight[0, 0])).to(device).repeat(bs, 1, 1)
#torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1000.00 GiB (GPU 0; 79.10 GiB total capacity; 42.02 GiB already allocated; 26.52 GiB free; 42.36 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF