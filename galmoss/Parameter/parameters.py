from galmoss.Parameter.basic import AbstractParameters

from typing import Union, Tuple
import numpy as np
import torch

class EffR(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    arc2pix: float = 0.1,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.05,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    new_scale = None if scale is None else [scale[0]/arc2pix, scale[1]/arc2pix]
    super().__init__(parameters= (parameters / arc2pix), scale=new_scale, step_length=step_length, if_fit=if_fit)   
    self.arc2pix = arc2pix
  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * self.arc2pix
  
  
  @property
  def FitValue(self):
    return self.fit_value.detach().cpu().numpy() * self.arc2pix
     

class CenX(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   
    
class CenY(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   
      
      
class PA(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    new_scale = None if scale is None else [scale[0] * torch.pi / 180, scale[1] * torch.pi / 180]
    super().__init__(parameters=parameters * torch.pi / 180, scale=new_scale, step_length=step_length, if_fit=if_fit)  

  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * 180 / np.pi  
  
  
  @property
  def FitValue(self):
    return self.fit_value.detach().cpu().numpy() * 180 / np.pi 
    
class Mag(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    Magnitude_zero: float = 0,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
    
  ):
    self.value_o = parameters
    new_scale = None if scale is None else [scale[0]-Magnitude_zero, scale[1]-Magnitude_zero]
    super().__init__(parameters=parameters-Magnitude_zero, scale = new_scale, step_length=step_length, if_fit=if_fit)   
    self.Magnitude_zero = Magnitude_zero
  @property
  def FitValue(self):
    return self.fit_value.detach().cpu().numpy() + self.Magnitude_zero


class SerN(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
    log: bool = True
    
  ):
    self.value_o = parameters 
    if log:
        super().__init__(parameters=np.log10(parameters), scale=(np.log10(scale[0]), np.log10(scale[1])), step_length=step_length, if_fit=if_fit, log=log)   
    else:
        super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit, log=log)        
    
class AxisR(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   
    
class Box(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = False,
  ):
    self.value_o = parameters
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)    
    

class SkyBg(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   
    
class Con(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   

class Fwhm(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    arc2pix: float = 0.1,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    new_scale = None if scale is None else [scale[0]/arc2pix, scale[1]/arc2pix]
    super().__init__(parameters= (parameters / arc2pix), scale=new_scale, step_length=step_length, if_fit=if_fit)   
    self.arc2pix = arc2pix

  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * self.arc2pix
  
  @property
  def FitValue(self):
    return self.fit_value.detach().cpu().numpy() * self.arc2pix
      
        
class TruncR(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    arc2pix: float = 1,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    new_scale = None if scale is None else [scale[0]/arc2pix, scale[1]/arc2pix]
    super().__init__(parameters= (parameters / arc2pix), scale=new_scale, step_length=step_length, if_fit=if_fit)   
    self.arc2pix = arc2pix
    
  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * self.arc2pix
   
  @property
  def FitValue(self):
    return self.fit_value.detach().cpu().numpy() * self.arc2pix
    
class TruncA(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)  
    
class TruncB(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)  
    
    
class Inten(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit) 
   
    
  
  
class Disp(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    arc2pix: float = 1,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    new_scale = None if scale is None else [scale[0]/arc2pix, scale[1]/arc2pix]
    
    super().__init__(parameters=(parameters/ arc2pix), scale=new_scale, step_length=step_length, if_fit=if_fit)   
    self.arc2pix = arc2pix
    
  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * self.arc2pix
  
class CoreR(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    arc2pix: float = 1,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    new_scale = None if scale is None else [scale[0]/arc2pix, scale[1]/arc2pix]
    
    super().__init__(parameters=(parameters/ arc2pix), scale=new_scale, step_length=step_length, if_fit=if_fit)   
    self.arc2pix = arc2pix
    
  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * self.arc2pix

class PowN(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 0.015,
    if_fit: bool = True,
  ):
    self.value_o = parameters
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit) 