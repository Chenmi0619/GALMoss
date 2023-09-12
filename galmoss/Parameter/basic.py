from typing import Union, Tuple
import numpy as np
import torch
from functools import wraps

device = 'cuda' if torch.cuda.is_available() else 'cpu'

  

      
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
    # self._effect_iter = torch.empty((0,))
    # self._chi_miu = torch.empty((0,)).to(device)
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
    step_length: float = 2.5e-2,
    # if_log: bool = False,
    if_fit: bool = True,
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
    

    if np.isscalar(parameters):
      self.parameters = torch.tensor([[parameters]]).to(device)
    elif isinstance(parameters, np.ndarray):
      self.parameters = torch.from_numpy(parameters).unsqueeze(0).to(device)
    elif isinstance(parameters, torch.Tensor):
      self.parameters = parameters.unsqueeze(0).to(device)
    else:
      assert "parameters should be torch.Tensor/ndarray vector if in parallel, a scalar float value if just fit one"
    
    
    
    self.param_length = self.parameters.shape[1]
    self.scale = scale
    self.proj_c = 1 if self.scale == None else (scale[1] - scale[0])/2 
    self.bestValue = None
    self.fitting_bestValue = None

    self.sq_bV = None
    self.step_length = step_length
  
    self.if_fit = if_fit

    self.uncertainty = torch.empty((0,)).to(device)
    self.fit_value = torch.empty((0,)).to(device)
    self.loop_value = torch.empty((0,)).to(device)
    self._loop_store = torch.empty((0,)).to(device)
    self.value_u = self.parameters
    
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
  
  

  @to_scale
  def value(self):
    return self.parameters.to(device)
  
  @to_scale
  def proj_fitted_value(self):
    # return self.fitting_bestValue.to(device)
    return self.fit_value.to(device)
    
  def Param_name(self):
    return 
  
  @to_scale
  def into(self, inputt):
    return inputt
  
  
  @inverse_scale
  # @inverse_log
  def transport_outof_optim(self, inputt) -> torch.Tensor: 
    """
    inputt:
      shape(batch_size)
    """
    return inputt
    

  # @property
  # def param_scale(self):
  #   if self.scale == None:
  #     return self.scale
  #   else:
  #     scale = torch.from_numpy(np.array(self.scale).astype(float)).to(device)
  #     if len(np.shape(scale)) == 1:
  #       scale = torch.unsqueeze(scale, dim=0)
  #     else:
  #       scale = scale
  #     return scale
  
  @property
  def steplength(self):
    return self.step_length
  

  def updating_v(self, mode="updating_value"):
    if mode == "updating_value":
      return self.value_b
    elif mode == "best_value":
      return self.b_bV
    else:
      assert "Wrong mode text, should be updating_value or best_value"
      
    
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
  def FitValue(self):
    return self.fit_value.detach().cpu().numpy() 
  
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
    self.fit_value = torch.concat((self.fit_value, fit_batch_value), dim=0)   
    


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
      
  

