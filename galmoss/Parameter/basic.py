from typing import Union, Tuple
import numpy as np
import torch
from functools import wraps
import numpy as np
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inv_from_log(func):

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
      
      
def proj_to_value_range(func):
    """
    To make the output to value_range space

    Parameters
    ----------
    func
        A function which returns the parameters need to be load in the optimism.

    Returns
    -------
        A function that returns the value_range-parameters need to be load in the optimism.
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
        value_range = cls.value_range
        
        if value_range is not None:  
          _value_range = (value_range[1] - value_range[0])/2
          _bias = 1 - value_range[1]/_value_range
          Params_value_range = (Params/ _value_range) + _bias
          Params_value_range = torch.atanh(Params_value_range)
          return Params_value_range
        else:
          return Params
        

    return wrapper

def invproj_from_value_range(func):
    """
    To make the output to value_range space

    Parameters
    ----------
    func
        A function which returns the parameters need to be load in the optimism.

    Returns
    -------
        A function that returns the value_range-parameters need to be load in the optimism.
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
        Params_value_range:
            shape: (batch_size)

        Returns
        -------
            The grid_like object whose coordinates are radially moved from (0.0, 0.0).
        """
        Params_value_range = func(cls, *args, **kwargs)
        Params_value_range = Params_value_range.to(device)
        value_range = cls.value_range #param_value_range
        
          
        if value_range is not None:  
          Params_value_range = torch.tanh(Params_value_range) 
          _bias = 1 - value_range[1]/cls.proj_c
          Params = (Params_value_range - _bias) * cls.proj_c
          return Params
        
        else:
          Params = Params_value_range
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
      A function that returns the value_range-parameters need to be load in the optimism.
  """
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, float], 
    value_range:  Union[Tuple[float, float], None] = None,
    step_length: float = 1.5,
    fit: bool = True,
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
    if value_range != None:
      assert ((parameters<=value_range[1] and parameters >= value_range[0]) if np.isscalar(parameters) 
              else (max(parameters)<=value_range[1] and min(parameters >= value_range[0]))), "Data out of range, please check!"
    
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
    self.value_range = value_range
    self.proj_c = 1 if self.value_range == None else (value_range[1] - value_range[0])/2 
    self.bestValue = None
    self.fitting_bestValue = None

    # self.sq_bV = None
    self.step_length = step_length

  
    self.fit = fit
    self.log = log
    
    self.uncertainty = torch.empty((0,)).to(device)
    self.Fitted_value_gpu = torch.empty((0,)).to(device)
    self.loop_value = torch.empty((0,)).to(device)
    self._loop_store = torch.empty((0,)).to(device)
    self.updating_value = self.parameters
    if log:
      self.initial_value = torch.pow(10, self.parameters)
    else:
      self.initial_value = self.parameters
    
    self.broadcasted_updating_value=None
    self.broadcasted_best_value=None
    """
    The parameter class.

    Parameters
    ----------
    parameters
        The input parameters.
    value_range
        If True, then will reflect the parameters into 0-1 field and use tanh function to value_range the parameters inside this value_range.
    if_log
        Some parameters will fit better in exp10 field.
    fit
        If this parameters need to be optimistic or just keep stable.
    """    
  
  
  @property
  @proj_to_value_range
  # @to_log  
  def proj_Ini_value(self):
    return self.parameters.to(device)
  
  @property
  @proj_to_value_range
  # @to_log
  def proj_Fitted_value(self):
    return self.Fitted_value_gpu.to(device)
    
  def Param_name(self):
    return 
  
  @proj_to_value_range
  # @to_log  
  def into(self, inputt):
    return inputt
  
  @inv_from_log
  @invproj_from_value_range
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
  #     return self.broadcasted_updating_value
  #   elif mode == "best_value":
  #     return self.broadcasted_best_value
  #   elif mode == "original":
  #     return self.initial_value    
  #   elif mode == "proj_Ini_value":
  #     return self.proj_Ini_value
  #   elif mode == "prof_fitted_value":
  #     return self.proj_Fitted_value
  #   else:
  #     assert "Wrong mode text, should be updating_value or best_value"
      
  def value(self, mode="updating_value"):
      mode_mapping = {
          "ub": self.broadcasted_updating_value,
          "u": self.updating_value,
          "bb": self.broadcasted_best_value,
          "i": self.initial_value,
          "proj_Ini_value": self.proj_Ini_value,
          "prof_fitted_value": self.proj_Fitted_value
      }
      
      if mode not in mode_mapping:
          raise ValueError("Wrong mode text, should be one of: updating_value, best_value, original, proj_Ini_value, prof_fitted_value")
      
      return mode_mapping[mode]
        
  def update(self, value):
    self.updating_value = self.transport_outof_optim(value).to(device) # updating_value

    
    self.broadcasted_updating_value = self.updating_value.unsqueeze(1).unsqueeze(2)


  def refresh_bv(self, index=None):
    if index == None:
      self.bestValue = self.updating_value
    else:
      self.bestValue[index] = self.updating_value[index]
    
    self.broadcasted_best_value = torch.unsqueeze(torch.unsqueeze(self.bestValue, 1), 1).detach()

  
  def grad_clean(self):
    self.grad = 0
    
  @property
  def Fitted_Value_cpu(self):
    return self.Fitted_value_gpu.detach().cpu().numpy() 
  
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
    

    if self.value_range is not None:  
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
      
  def changing_fitting(self, fit):
    self.fit = fit

  def __repr__(self) -> str:
     return self.__class__.__name__
      
  

