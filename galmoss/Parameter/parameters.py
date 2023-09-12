from galmoss.Parameter.basic import AbstractParameters

from typing import Union, Tuple
import numpy as np
import torch

class Effective_radius(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    arc2pix: float = 1,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    new_scale = None if scale is None else [scale[0]/arc2pix, scale[1]/arc2pix]
    super().__init__(parameters=np.divide(parameters, arc2pix), scale=new_scale, step_length=step_length, if_fit=if_fit)   
    self.arc2pix = arc2pix
  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * self.arc2pix
  
  
  @property
  def FitValue(self):
    return self.fit_value.detach().cpu().numpy() * self.arc2pix
     

class Centre_x(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   
    
class Centre_y(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   
      
class Inclination(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    new_scale = None if scale is None else [scale[0] * torch.pi / 180, scale[1] * torch.pi / 180]
    super().__init__(parameters=parameters * torch.pi / 180, scale=new_scale, step_length=step_length, if_fit=if_fit)  

  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * 180 / np.pi  
  
  
  @property
  def FitValue(self):
    return self.fit_value.detach().cpu().numpy() * 180 / np.pi 
    
class Magnitude(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    Magnitude_zero: float = 0,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
    
  ):
    new_scale = None if scale is None else [scale[0]-Magnitude_zero, scale[1]-Magnitude_zero]
    super().__init__(parameters=parameters-Magnitude_zero, scale = new_scale, step_length=step_length, if_fit=if_fit)   
    self.Magnitude_zero = Magnitude_zero
  @property
  def FitValue(self):
    return self.fit_value.detach().cpu().numpy() + self.Magnitude_zero


class Sersic_index(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
    
    
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   

    
class Axis_ratio(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   
    
class Boxness(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = False,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)    
    

class Sky_background(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   
    
class Concentration(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   

class Fwhm(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)   
    
    
class Truncation_radius(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    arc2pix: float = 1,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=np.divide(parameters, arc2pix), scale=[scale[0]/arc2pix, scale[1]/arc2pix], step_length=step_length, if_fit=if_fit)   
    self.arc2pix = arc2pix
  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * self.arc2pix
   
  @property
  def FitValue(self):
    return self.fit_value.detach().cpu().numpy() * self.arc2pix
    
class Truncation_strength(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)  
    
class Powerlaw_slope(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit)  
    
    
class Intensity(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit) 
   
    
class Scale_length(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    arc2pix: float = 1,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    new_scale = None if scale is None else [scale[0]/arc2pix, scale[1]/arc2pix]
    
    super().__init__(parameters=np.divide(parameters, arc2pix), scale=new_scale, step_length=step_length, if_fit=if_fit)   
    self.arc2pix = arc2pix
    
  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * self.arc2pix
  
  
class Dispersion(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    arc2pix: float = 1,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    new_scale = None if scale is None else [scale[0]/arc2pix, scale[1]/arc2pix]
    
    super().__init__(parameters=np.divide(parameters, arc2pix), scale=new_scale, step_length=step_length, if_fit=if_fit)   
    self.arc2pix = arc2pix
    
  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * self.arc2pix
  
class Core_radius(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    arc2pix: float = 1,
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    new_scale = None if scale is None else [scale[0]/arc2pix, scale[1]/arc2pix]
    
    super().__init__(parameters=np.divide(parameters, arc2pix), scale=new_scale, step_length=step_length, if_fit=if_fit)   
    self.arc2pix = arc2pix
    
  @property
  def Uncertainty(self):
    return self.uncertainty.detach().cpu().numpy() * self.arc2pix

class Powerlaw_index(AbstractParameters):
  def __init__(
    self, 
    parameters: Union[np.ndarray, torch.Tensor, Tuple[float, float]], 
    scale:  Union[np.ndarray, torch.Tensor, Tuple[float, float], None] = None,
    step_length: float = 1.5e-2,
    if_fit: bool = True,
  ):
    super().__init__(parameters=parameters, scale=scale, step_length=step_length, if_fit=if_fit) 