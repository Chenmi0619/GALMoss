from collections.abc import Sequence as Seq
from typing import Union, Tuple
import numpy as np
import torch
from functools import wraps
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_log(func):
    """
    To make the output to log space

    Parameters
    ----------
    func
        A function which returns the parameters need to be load in the optimism.

    Returns
    -------
        A function that returns the log-parameters need to be load in the optimism.
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

        if_log = cls.if_log
        
        if if_log == True:
          Params_log10 = torch.log10(Params)
          return Params_log10
        else:
          return Params
        

    return wrapper

def inverse_log(func):
    """
    To make the input from log space to linear space

    Parameters
    ----------
    func
        A function which returns the parameters in the optimism to the normal output.

    Returns
    -------
        A function that returns the normal-parameters need to be output.
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
        if_log = cls.if_log
        
        if if_log == True:
          Params_pow10 = torch.pow(10, Params)
          return Params_pow10
        else:
          return Params

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
        scale = cls.param_scale
        
        
          
        if scale is not None:  

          _scale = (scale[:, 1] - scale[:, 0])/2
          _bias = 1 - scale[:, 1]/_scale
          Params_scale = (Params/ _scale) + _bias
          
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

        Returns
        -------
            The grid_like object whose coordinates are radially moved from (0.0, 0.0).
        """
        Params_scale = func(cls, *args, **kwargs)
        scale = cls.param_scale
        log = cls.if_log
        
          
        if scale is not None:  
          # if log is True:
          #   scale = torch.log10(scale)
          # else:
          #   scale = scale
            
          _scale = (scale[:, 1] - scale[:, 0])/2
          cls.grad_C = _scale
          _bias = 1 - scale[:, 1]/_scale
          Params = (Params_scale - _bias) * _scale
          # print("_scale", _scale)
          return Params
        
        else:
          return Params_scale

    return wrapper
          
class AbstractParameters:
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
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 2.5e-2,
    if_log: bool = False,
    if_fit: bool = True,
  ):
    if len(np.shape(torch.from_numpy(np.array(parameters).astype(float)))) < 2:
      self.paramters = torch.unsqueeze(torch.from_numpy(np.array(parameters).astype(float)), dim=0)
    else:
      self.paramters = torch.from_numpy(parameters)
    

    self.scale = scale
    self.grad_C = None
    self.step_length = step_length
  
    self.if_log = if_log
    self.if_fit = if_fit
    self.stock = []
    self.jacob = []
    self.grad = 0
    self.uncertainty = torch.Tensor([]).to(device)
    self.fit_value = torch.Tensor([]).to(device)
    self.update_value = None
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
  
  
  @to_log
  @to_scale
  def value(self):
    return self.paramters.to(device)
  
  def Param_name(self):
    return 
  
  
  @inverse_scale
  @inverse_log
  def transport_outof_optim(self, inputt):
    return inputt
    
  
  @property
  @to_log
  def param_scale(self):
    if self.scale == None:
      return self.scale
    else:
      scale = torch.from_numpy(np.array(self.scale).astype(float)).to(device)
      if len(np.shape(scale)) == 1:
        scale = torch.unsqueeze(scale, dim=0)
      else:
        scale = scale
      return scale
  
  @property
  def steplength(self):
    return self.step_length
  
  def update(self, value):
    self.param_store(self.transport_outof_optim(value))
    self.update_value = self.transport_outof_optim(value)
    self.sq_V = torch.unsqueeze(torch.unsqueeze(self.transport_outof_optim(value), 1), 1)
    
  
  def cal_uncertainty(self, loss, aaa):
    index = torch.where(aaa == 0)
    # aa = torch.pow(1/aaa, 2)
    aa = 1/aaa
    aa[index] = 0
    print(np.shape(aaa))
    print(np.shape(loss))
    loss2 = loss * aa
    # print(loss)
    # print(np.shape(loss), np.shape(aaa), np.shape(loss2))
    loss3 = torch.sum(loss2)
    # print("-"*20)
    # print(torch.autograd.grad(loss[0][0][0], self.update_value, retain_graph=True))
    # print(torch.autograd.grad(loss[0][0][1], self.update_value, retain_graph=True))
    # print("-"*20)
    grad1 = torch.autograd.grad(loss3, self.update_value, retain_graph=True)
    grad2 = torch.autograd.grad(torch.sum(loss), self.update_value, retain_graph=True)
    # print(grad1)
    # print(dd)
    # print(grad2)
    ccc = torch.pow(1/(grad1[0]*grad1[0]), 0.5)
    # print(grad1[0], grad1[0]*grad1[0])
    # print(ccc)
    # print(self.update_value)
    return (grad1[0]*grad1[0])

    # print((torch.sum(aa, dim=[1, 2])))
    # print( self.update_value)
    # Print = torch.tensor([]).to(device)
    # for anygrad in grad1[0]:  # torch.autograd.grad返回的是元组
    #     Print = torch.cat((Print, torch.autograd.grad(anygrad, self.update_value, retain_graph=True)[0]))
    # print(np.shape(Print))
    # print(Print)
        
    # print(Print.view(self.update_value.size()[0], -1))

  def param_store(self, aaa):
    self.stock.append(aaa)
  
  def grad_store(self, aaa):
    self.grad = self.grad + aaa
  
  def grad_clean(self):
    self.grad = 0
    
  def uncertainty_store(self, aaa):
    if len(self.uncertainty) == 0:
      self.uncertainty = aaa
    else:
      self.uncertainty = torch.concat((
        self.uncertainty, aaa), dim=0)

  def value_store(self, aaa):
    if len(self.fit_value) == 0:
      self.fit_value = aaa
    else:
      self.fit_value = torch.concat((
        self.fit_value, aaa), dim=0)
      
  def __repr__(self) -> str:
     return self.__class__.__name__
      
  

# if __name__ == "__main__":
#   a = [2]
#   aa = np.array(a)
#   index_n = Parameters([2], if_log=False)
#   print(index_n.__class__.__name__)

