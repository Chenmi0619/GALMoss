from typing import Union, Tuple
import numpy as np
import torch
from functools import wraps
import numpy as np
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def to_log(func):
#     """
#     To make the output to log space

#     Parameters
#     ----------
#     func
#         A function which returns the parameters need to be load in the optimism.

#     Returns
#     -------
#         A function that returns the log-parameters need to be load in the optimism.
#     """
#     @wraps(func)
#     def wrapper(
#         cls,
#         *args,
#         **kwargs
#     ) -> np.ndarray:

#         Params = func(cls, *args, **kwargs)
        
#         if cls.log:  
#           print("torch.log10", torch.log10(Params))
#           return torch.log10(Params)
#         else:
#           return Params
        
#     return wrapper

def inverse_log(func):

    @wraps(func)
    def wrapper(
        cls,
        *args,
        **kwargs
    ) -> np.ndarray:
        
        Params_log = func(cls, *args, **kwargs)
        # Params_log = Params_log.to(device)
       
        if cls.log:  
          return torch.pow(10, Params_log)   
            
        else:
          return Params_log
        
      
    return wrapper
      

      
def to_scale(func):
    """
    To make the output to scale space

    Parameters
    ----------
    func
        A function which returns the parameters need to be load in the optimism.

    Returns
    -------
        A function that returns the scale-parameters need to be load in the optimism.
    """
    @wraps(func)
    def wrapper(
        cls,
        *args,
        **kwargs
    ) -> np.ndarray:
        """

        Parameters
        ----------
        cls : Parameters
            The class that owns the function.
        Params :
            The (y, x) coordinates which are to be radially moved from (0.0, 0.0).

        Returns
        -------
            The grid_like object whose coordinates are radially moved from (0.0, 0.0).
        """
        Params = func(cls, *args, **kwargs)
        scale = cls.scale
        
        if scale is not None:  
          _scale = (scale[1] - scale[0])/2
          _bias = 1 - scale[1]/_scale
          Params_scale = (Params/ _scale) + _bias
          Params_scale = torch.atanh(Params_scale)
          return Params_scale
        else:
          return Params
        

    return wrapper

def inverse_scale(func):
    """
    To make the output to scale space

    Parameters
    ----------
    func
        A function which returns the parameters need to be load in the optimism.

    Returns
    -------
        A function that returns the scale-parameters need to be load in the optimism.
    """
    @wraps(func)
    def wrapper(
        cls,
        *args,
        **kwargs
    ) -> np.ndarray:
        """

        Parameters
        ----------
        cls : Parameters
            The class that owns the function.
        Params :
            The (y, x) coordinates which are to be radially moved from (0.0, 0.0).
        Params_scale:
            shape: (batch_size)

        Returns
        -------
            The grid_like object whose coordinates are radially moved from (0.0, 0.0).
        """
        Params_scale = func(cls, *args, **kwargs)
        Params_scale = Params_scale.to(device)
        scale = cls.scale #param_scale
        
          
        if scale is not None:  
          Params_scale = torch.tanh(Params_scale) 
          _bias = 1 - scale[1]/cls.proj_c
          Params = (Params_scale - _bias) * cls.proj_c
          return Params
        
        else:
          Params = Params_scale
          return Params
        
      
    return wrapper

class ParametersRepo:
  def __init__(self):
    self._loop_store_bs = torch.empty((0,)).to(device)
    self._loop_store_total = torch.empty((0,)).to(device)
           
class AbstractParameters(ParametersRepo):
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
    parameters: Union[np.ndarray, torch.Tensor, float], 
    scale:  Union[Tuple[float, float], None] = None,
    step_length: float = 1.5,
    # if_log: bool = False,
    if_fit: bool = True,
    log: bool = False
  ):
    """
    parameters: 
      shape: (n_galaxy)
    self.parameters:
      shape: (1, n_galaxy)
    """
    super().__init__()
    assert (np.isscalar(parameters) or len(parameters.shape) == 1), "Input a must be a 1-dimensional torch.Tensor/ndarray vector if in parallel, a scalar float value if just fit one"
    # rest of your code here
    if scale != None:
      assert ((parameters<=scale[1] and parameters >= scale[0]) if np.isscalar(parameters) 
              else (max(parameters)<=scale[1] and min(parameters >= scale[0]))), "Data out of range, please check!"
    
    print("a")
    if np.isscalar(parameters):
      self.parameters = torch.tensor([[parameters]]).to(device)
    elif isinstance(parameters, np.ndarray):
      self.parameters = torch.from_numpy(parameters).squeeze().to(device)
    elif isinstance(parameters, torch.Tensor):
      self.parameters = parameters.squeeze().to(device)
    elif isinstance(parameters, pd.Series):
      self.parameters = torch.from_numpy(parameters.values).squeeze().to(device)
    else:
      assert False, "parameters should be torch.Tensor/ndarray vector/pandas.series if in parallel, a scalar float value if just fit one"

    print("self.parameters.shape", self.parameters.shape)
    
    self.param_length = self.parameters.shape[0]
    self.scale = scale
    self.proj_c = 1 if self.scale == None else (scale[1] - scale[0])/2 
    self.bestValue = None
    self.fitting_bestValue = None

    # self.sq_bV = None
    self.step_length = step_length

  
    self.if_fit = if_fit
    self.log = log
    
    self.uncertainty = torch.empty((0,)).to(device)
    self.Fitted_value_gpu = torch.empty((0,)).to(device)
    self.loop_value = torch.empty((0,)).to(device)
    self._loop_store = torch.empty((0,)).to(device)
    self.value_u = self.parameters
    if log:
      self.value_o = torch.pow(10, self.parameters)
    else:
      self.value_o = self.parameters
    
    self.value_b=None
    self.b_bV=None
    """
    The parameter class.

    Parameters
    ----------
    parameters
        The input parameters.
    scale
        If True, then will reflect the parameters into 0-1 field and use tanh function to scale the parameters inside this scale.
    if_log
        Some parameters will fit better in exp10 field.
    if_fit
        If this parameters need to be optimistic or just keep stable.
    """    
  
  
  @property
  @to_scale
  # @to_log  
  def proj_Ini_value(self):
    return self.parameters.to(device)
  
  @property
  @to_scale
  # @to_log
  def proj_Fitted_value(self):
    return self.Fitted_value_gpu.to(device)
    
  def Param_name(self):
    return 
  
  @to_scale
  # @to_log  
  def into(self, inputt):
    return inputt
  
  @inverse_log
  @inverse_scale
  def transport_outof_optim(self, inputt) -> torch.Tensor: 
    """
    inputt:
      shape(batch_size)
    """
    return inputt
    
  
  @property
  def steplength(self):
    return self.step_length
  

  # def value(self, mode="updating_value"):
  #   if mode == "updating_value":
  #     return self.value_b
  #   elif mode == "best_value":
  #     return self.b_bV
  #   elif mode == "original":
  #     return self.value_o    
  #   elif mode == "proj_Ini_value":
  #     return self.proj_Ini_value
  #   elif mode == "prof_fitted_value":
  #     return self.proj_Fitted_value
  #   else:
  #     assert "Wrong mode text, should be updating_value or best_value"
      
  def value(self, mode="updating_value"):
      mode_mapping = {
          "updating_value": self.value_b,
          "best_value": self.b_bV,
          "original": self.value_o,
          "proj_Ini_value": self.proj_Ini_value,
          "prof_fitted_value": self.proj_Fitted_value
      }
      
      if mode not in mode_mapping:
          raise ValueError("Wrong mode text, should be one of: updating_value, best_value, original, proj_Ini_value, prof_fitted_value")
      
      return mode_mapping[mode]
        
  def update(self, value):
    self.value_u = self.transport_outof_optim(value).to(device) # value_u
    self.value_b = self.value_u.unsqueeze(1).unsqueeze(2)
    

  def refresh_bv(self, index=None):
    if index == None:
      self.bestValue = self.value_u
    else:
      self.bestValue[index] = self.value_u[index]
    
    self.b_bV = torch.unsqueeze(torch.unsqueeze(self.bestValue, 1), 1).detach()

  
  def grad_clean(self):
    self.grad = 0
    
  @property
  def Fitted_Value_cpu(self):
    return self.fitted_value.detach().cpu().numpy() 
  
  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() 
  
  def uncertainty_store(self, proj_uc):
    """
    This function is for calculating the final uncertainty of parameters. Inside function "cm_uncertainty", we get the uncertianty of projected parameters, so we need the error propagation equation to expand.
    
    uncertainty:
      shape: (cm_batch_size, 1)
    self.uncertainty:
      shape:(n_galaxy)
    """    
    

    if self.scale is not None:  
      real_uc = self.proj_c * (1 - torch.pow(torch.tanh(self.into(self.bestValue)), 2)) * proj_uc
    else:
      real_uc = proj_uc


    self.uncertainty = torch.concat((self.uncertainty, real_uc), dim=0)

  def value_store(self, fit_batch_value):
    self.Fitted_value_gpu = torch.concat((self.Fitted_value_gpu, fit_batch_value), dim=0)   
    


  def loop_store(self, fit_value):
    if self.dataset.mode == "bsp_uc":
      loop_repo = self.bootstrap_value 
    elif self.dataset.mode == "wr_uc":
      loop_repo = self.resample_value 
    loop_repo = torch.concat((
        loop_repo, fit_value), dim=1)
      
  def changing_fitting(self, if_fit):
    self.if_fit = if_fit

  def __repr__(self) -> str:
     return self.__class__.__name__
      
  

