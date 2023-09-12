from galmoss.profile.ellip_profile import Ellip
from galmoss.Parameter.parameters import Centre_x, Centre_y, Inclination, Axis_ratio, Boxness, Effective_radius, Magnitude, Intensity, Sersic_index, Concentration, Fwhm, Sky_background, Truncation_radius, Truncation_strength, Powerlaw_slope, Scale_length, Dispersion, Core_radius, Powerlaw_index
from typing import Union, Tuple
# from fitting import Fitting
import pynvml

from functools import wraps
import numpy as np
import torch
import inspect
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# def control_mode(func):
#     """
#     to control the grad
#     """
#     @wraps(func)
#     def wrapper(
#         cls,
#         radius,
#         output,
#         *args, 
#         **kwargs
#     ):
#         if output:
#             # with torch.no_grad():
#             #     mode = "best_value_broadcast"
#             #     print("dddd")
#           return func(cls, radius, output, mode = "best_value_broadcast", *args, **kwargs)
#         else:
#           return func(cls, radius, output, *args, **kwargs)
#     return wrapper
  
class Light_Profile:
  def image_2d_grid_from_(self, grid, mode="updating_value"):
    return self.image_2d_from_(self.make_radius(grid, mode))  
    
  

class Sky(Light_Profile):
  def __init__(
  self, 
  sky_background:Sky_background

):
    self.sky_background = sky_background
  
 
  def image_2d_from_(self, radius, mode="updating_value"):
    return torch.ones_like(radius) * self.sky_background.updating_v(mode)
              
  

  def param(self, name: str) -> dict: 
    flex_dic = {}
    fix_dic = {}
    test_dic = {}
    
    for param in [self.sky_background]:
      if id(param) not in test_dic:
        value = name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = value
        if param.if_fit == True:
          flex_dic[value] = param
        else:
          fix_dic[value] = param
                
      else:
        _value = test_dic[id(param)]
        _new_value = _value + "  and  " + name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = _new_value
    
        if param.if_fit == True:
          flex_dic[_new_value] = flex_dic.pop(_value)
        else:
          fix_dic[_new_value] = flex_dic.pop(_value)
    return flex_dic, fix_dic, test_dic
    

  @classmethod
  def definition(
    cls,
    sky_background: Sky_background, 
    
  ) -> "Sky":
      """
      """
      length = []
      args, _, _, values = inspect.getargvalues(inspect.currentframe())
      
      return Sky(
            sky_background=sky_background
      )
      
class Sersic(Ellip, Light_Profile):
  def __init__(
  self, 
  centre_x: Centre_x, 
  centre_y: Centre_y, 
  inclination: Inclination,
  axis_ratio: Axis_ratio,
  
  effective_radius: Effective_radius,
  magnitude: Magnitude,
  sersic_index: Sersic_index,
  
  boxness: Boxness = None,

):
    super().__init__(centre_x=centre_x, centre_y=centre_y, inclination=inclination, axis_ratio=axis_ratio, boxness=boxness)
    self.effective_radius = effective_radius
    self.magnitude = magnitude
    self.sersic_index = sersic_index 
    

  def sersic_constant(self, sersic_index) -> float:
    """
    A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
    total integrated light.
    """

    return (
        (2 * sersic_index)
        - (1.0 / 3.0)
        + (4.0 / (405.0 * sersic_index))
        + (46.0 / (25515.0 * sersic_index))
        + (131.0 / (1148175.0 * sersic_index))
        - (2194697.0 / (30690717750.0 * sersic_index))
    )


  
    
  def image_2d_from_(self, radius, mode="updating_value"):
    l_tot = torch.pow(self.effective_radius.updating_v(mode), 2) * 2 * torch.pi * self.sersic_index.updating_v(mode) * torch.exp(self.sersic_constant(self.sersic_index.updating_v(mode))) * self.axis_ratio.updating_v(mode) * torch.pow(10, torch.log10(torch.exp(torch.lgamma(2 * self.sersic_index.updating_v(mode)))) - 2 * self.sersic_index.updating_v(mode) * torch.log10(self.sersic_constant(self.sersic_index.updating_v(mode))))
    
    F_tot = torch.pow(10, (-0.4 * self.magnitude.updating_v(mode))) * self.R_box(mode)  
   
    return (F_tot/l_tot) * torch.exp(-self.sersic_constant(self.sersic_index.updating_v(mode))
                          * ( torch.pow(radius / self.effective_radius.updating_v(mode), (1.0 / self.sersic_index.updating_v(mode))) - 1))

                    
  
  def image_2d_grid_from_(self, grid, mode="updating_value"):
    return self.image_2d_from_(self.make_radius(grid, mode), mode)
  

  def param(self, name: str) -> dict: 
    flex_dic = {}
    fix_dic = {}
    test_dic = {}
    
    for param in [self.centre_x, self.centre_y, self.inclination, self.axis_ratio, 
                  self.effective_radius, self.magnitude, self.sersic_index, self.boxness]:
      if id(param) not in test_dic:
        value = name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = value
        if param.if_fit == True:
          flex_dic[value] = param
        else:
          fix_dic[value] = param
                
      else:
        _value = test_dic[id(param)]
        _new_value = _value + "  and  " + name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = _new_value
    
        if param.if_fit == True:
          flex_dic[_new_value] = flex_dic.pop(_value)
        else:
          fix_dic[_new_value] = flex_dic.pop(_value)
    return flex_dic, fix_dic, test_dic


  @classmethod
  def definition(
    cls,
    centre_x: Centre_x, 
    centre_y: Centre_y, 
    inclination: Inclination,
    axis_ratio: Axis_ratio,
    
    effective_radius: Effective_radius,
    magnitude: Magnitude,
    sersic_index: Sersic_index,
    
    boxness: Boxness = None,
  ) -> "Sersic":
      """
      """
      length = []
      args, _, _, values = inspect.getargvalues(inspect.currentframe())
      if boxness == None:
        boxness = Boxness(parameters=torch.zeros(length[0]))

      for _, arg in enumerate(args[1: ]):
        length.append(values[arg].param_length)
      assert all(x == length[0] for x in length), "the length of parameters are not the same!"
        
      return Sersic(
            centre_x=centre_x, 
            centre_y=centre_y, 
            inclination=inclination,
            axis_ratio=axis_ratio,
            
            effective_radius=effective_radius,
            magnitude=magnitude,
            sersic_index=sersic_index,
            
            boxness=boxness,
      )
      
      
class Moffat(Ellip, Light_Profile):
  def __init__(
  self, 
  centre_x: Centre_x, 
  centre_y: Centre_y, 
  inclination: Inclination,
  axis_ratio: Axis_ratio,
  
  magnitude: Magnitude,
  concentration: Concentration,
  fwhm: Fwhm,
  
  boxness: Boxness,
):
    super().__init__(centre_x=centre_x, centre_y=centre_y, inclination=inclination, axis_ratio=axis_ratio, boxness=boxness)
    self.magnitude = magnitude
    self.concentration = concentration
    self.fwhm = fwhm
    
    
  def image_2d_from_(self, radius, mode="updating_value"):
    Rd = self.fwhm.updating_v(mode) / (2 * torch.pow(torch.pow(2, 1/self.concentration.updating_v(mode)) -1, 0.5))

    l_tot = (torch.pi * self.axis_ratio.updating_v(mode) * torch.pow(Rd, 2)) / (self.concentration.updating_v(mode) -1)
    
    F_tot = torch.pow(10, (-0.4 * self.magnitude.updating_v(mode))) * self.R_box(mode)
    return (F_tot/l_tot) * torch.pow(1 + 
                                     torch.pow(radius/Rd, 2),
                                     -self.concentration.updating_v(mode)                    
    )
                    
  

  

  def param(self, name: str) -> dict: 
    flex_dic = {}
    fix_dic = {}
    test_dic = {}
    
    for param in [self.centre_x, self.centre_y, self.inclination, self.axis_ratio, self.magnitude, self.concentration, self.fwhm, self.boxness]:
      if id(param) not in test_dic:
        value = name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = value
        if param.if_fit == True:
          flex_dic[value] = param
        else:
          fix_dic[value] = param
                
      else:
        _value = test_dic[id(param)]
        _new_value = _value + "  and  " + name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = _new_value
    
        if param.if_fit == True:
          flex_dic[_new_value] = flex_dic.pop(_value)
        else:
          fix_dic[_new_value] = flex_dic.pop(_value)
    return flex_dic, fix_dic, test_dic
    

    

  @classmethod
  def definition(
    cls,
    centre_x: Centre_x, 
    centre_y: Centre_y, 
    inclination: Inclination,
    axis_ratio: Axis_ratio,
    
    magnitude: Magnitude,
    concentration: Concentration,
    fwhm: Fwhm,
    
    boxness: Boxness = None,
    
    
  ) -> "Moffat":
      """
      """
      length = []
      args, _, _, values = inspect.getargvalues(inspect.currentframe())
      if boxness == None:
        boxness = Boxness(parameters=torch.zeros(length[0]))

      for _, arg in enumerate(args[1: ]):
        length.append(values[arg].param_length)
      assert all(x == length[0] for x in length), "the length of parameters are not the same!"
      
      return Moffat(
            centre_x=centre_x, 
            centre_y=centre_y, 
            inclination=inclination,
            axis_ratio=axis_ratio,
            
            magnitude=magnitude,
            concentration=concentration,
            fwhm=fwhm,
            
            boxness=boxness,
      )


class ExpDisk(Ellip, Light_Profile):
  def __init__(
  self, 
  centre_x: Centre_x, 
  centre_y: Centre_y, 
  inclination: Inclination,
  axis_ratio: Axis_ratio,
  
  magnitude: Magnitude,
  scale_length: Scale_length,
  
  boxness: Boxness


):
    super().__init__(centre_x=centre_x, centre_y=centre_y, inclination=inclination, axis_ratio=axis_ratio, boxness=boxness)
    self.scale_length = scale_length
    self.magnitude = magnitude
 

  def image_2d_from_(self, radius, output=False, mode="updating_value_broadcast"):

    l_tot = 2 * torch.pi * self.axis_ratio.updating_v(mode) * torch.pow(self.scale_length.updating_v(mode), 2)
    F_tot = torch.pow(10, (-0.4 * self.magnitude.updating_v(mode))) * self.R_box(mode)  
        
    return (F_tot/l_tot) * torch.exp(-radius/self.scale_length.updating_v(mode))
                    
  
  def image_2d_grid_from_(self, grid, output=False):
    return self.image_2d_from_(self.make_radius(grid, output), output)
  

  def param(self, name: str) -> dict: 
    flex_dic = {}
    fix_dic = {}
    test_dic = {}

    for param in [self.centre_x, self.centre_y, self.inclination, self.axis_ratio, self.magnitude, self.scale_length, self.boxness]:
      if id(param) not in test_dic:
        value = name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = value
        if param.if_fit == True:
          flex_dic[value] = param
        else:
          fix_dic[value] = param
                
      else:
        _value = test_dic[id(param)]
        _new_value = _value + "  and  " + name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = _new_value
    
        if param.if_fit == True:
          flex_dic[_new_value] = flex_dic.pop(_value)
        else:
          fix_dic[_new_value] = flex_dic.pop(_value)
    return flex_dic, fix_dic, test_dic

      
  @classmethod
  def definition(
    cls,
    centre_x: Centre_x, 
    centre_y: Centre_y, 
    inclination: Inclination,
    axis_ratio: Axis_ratio,
    
    
    magnitude: Magnitude,
    scale_length: Scale_length,
    
    boxness: Boxness = None,

  ) -> "ExpDisk":
      """
      """
      length = []
      args, _, _, values = inspect.getargvalues(inspect.currentframe())
      if boxness == None:
        boxness = Boxness(parameters=torch.zeros(length[0]))

      for _, arg in enumerate(args[1: ]):
        length.append(values[arg].param_length)
      assert all(x == length[0] for x in length), "the length of parameters are not the same!"
      
      return ExpDisk(
            centre_x=centre_x, 
            centre_y=centre_y, 
            inclination=inclination,
            axis_ratio=axis_ratio,
            
            magnitude=magnitude,
            scale_length=scale_length,
            boxness=boxness
      )      
      

class Gaussian(Ellip, Light_Profile):
  def __init__(
  self, 
  centre_x: Centre_x, 
  centre_y: Centre_y, 
  inclination: Inclination,
  axis_ratio: Axis_ratio,
  
  intensity: Intensity,
  dispersion: Dispersion,
  boxness: Boxness


):
    super().__init__(centre_x=centre_x, centre_y=centre_y, inclination=inclination, axis_ratio=axis_ratio, boxness=boxness)
    self.dispersion = dispersion
    self.intensity = intensity
 

  def image_2d_from_(self, radius, mode="updating_value"):

    # l_tot = 2 * torch.pi * self.axis_ratio.updating_v(mode) * torch.pow(self.dispersion.updating_v(mode))
    # F_tot = torch.pow(10, (-0.4 * self.magnitude.updating_v(mode))) * self.R_box(mode)  
        
    return self.intensity.updating_v(mode) * torch.exp(-torch.pow(radius, 2)/(2 * torch.pow(self.dispersion.updating_v(mode), 2)))
                    
  
  def image_2d_grid_from_(self, grid, mode="updating_value"):
    return self.image_2d_from_(self.make_radius(grid, mode), mode)
  

  def param(self, name: str) -> dict: 
    flex_dic = {}
    fix_dic = {}
    test_dic = {}

    for param in [self.centre_x, self.centre_y, self.inclination, self.axis_ratio, self.intensity, self.dispersion, self.boxness]:
      if id(param) not in test_dic:
        value = name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = value
        if param.if_fit == True:
          flex_dic[value] = param
        else:
          fix_dic[value] = param
                
      else:
        _value = test_dic[id(param)]
        _new_value = _value + "  and  " + name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = _new_value
    
        if param.if_fit == True:
          flex_dic[_new_value] = flex_dic.pop(_value)
        else:
          fix_dic[_new_value] = flex_dic.pop(_value)
    return flex_dic, fix_dic, test_dic

      
  @classmethod
  def definition(
    cls,
    centre_x: Centre_x, 
    centre_y: Centre_y, 
    inclination: Inclination,
    axis_ratio: Axis_ratio,
    
    intensity: Intensity,
    dispersion: Dispersion,
    boxness: Boxness = None,

  ) -> "Gaussian":
      """
      """
      length = []
      args, _, _, values = inspect.getargvalues(inspect.currentframe())
      if boxness == None:
        boxness = Boxness(parameters=torch.zeros(length[0]))

      for _, arg in enumerate(args[1: ]):
        length.append(values[arg].param_length)
      assert all(x == length[0] for x in length), "the length of parameters are not the same!"
      
      return Gaussian(
            centre_x=centre_x, 
            centre_y=centre_y, 
            inclination=inclination,
            axis_ratio=axis_ratio,
            
            intensity=intensity,
            dispersion=dispersion,
            boxness=boxness
      )      
      
      
    
       
    
              
class ModifiedFerrer(Ellip, Light_Profile):
  def __init__(
  self, 
  centre_x: Centre_x, 
  centre_y: Centre_y, 
  inclination: Inclination,
  axis_ratio: Axis_ratio,
  
  
  intensity: Intensity,
  truncation_radius: Truncation_radius,
  truncation_strength: Truncation_strength,
  powerlaw_slope: Powerlaw_slope,
  
  boxness: Boxness

):
    super().__init__(centre_x=centre_x, centre_y=centre_y, inclination=inclination, axis_ratio=axis_ratio, boxness=boxness)
    self.intensity = intensity
    self.truncation_radius = truncation_radius
    self.truncation_strength = truncation_strength
    self.powerlaw_slope = powerlaw_slope

  def image_2d_from_(self, radius, mode="updating_value"):
    return self.intensity.updating_v(mode) * torch.pow(1 - 
                                    torch.pow(radius/self.truncation_radius.updating_v(mode), 2 - self.truncation_strength.updating_v(mode)),
                                    self.powerlaw_slope.updating_v(mode))
                    
  
  def image_2d_grid_from_(self, grid, mode="updating_value"):
    return self.image_2d_from_(self.make_radius(grid, mode), mode)
  

  def param(self, name: str) -> dict: 
    flex_dic = {}
    fix_dic = {}
    test_dic = {}
    
    for param in [self.centre_x, self.centre_y, self.inclination, self.axis_ratio, self.intensity, self.truncation_radius, self.truncation_strength, self.powerlaw_slope, self.boxness]:
      if id(param) not in test_dic:
        value = name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = value
        if param.if_fit == True:
          flex_dic[value] = param
        else:
          fix_dic[value] = param
                
      else:
        _value = test_dic[id(param)]
        _new_value = _value + "  and  " + name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = _new_value
    
        if param.if_fit == True:
          flex_dic[_new_value] = flex_dic.pop(_value)
        else:
          fix_dic[_new_value] = flex_dic.pop(_value)
    return flex_dic, fix_dic, test_dic

      
  @classmethod
  def definition(
    cls,
    centre_x: Centre_x, 
    centre_y: Centre_y, 
    inclination: Inclination,
    axis_ratio: Axis_ratio,
    
    intensity: Intensity,
    truncation_radius: Truncation_radius,
    truncation_strength: Truncation_strength,
    powerlaw_slope: Powerlaw_slope,
    
    boxness: Boxness = None,
  ) -> "ModifiedFerrer":
      """
      """
      length = []
      args, _, _, values = inspect.getargvalues(inspect.currentframe())
      if boxness == None:
        boxness = Boxness(parameters=torch.zeros(length[0]))

      for _, arg in enumerate(args[1: ]):
        length.append(values[arg].param_length)
      assert all(x == length[0] for x in length), "the length of parameters are not the same!"
      
      return ModifiedFerrer(
            centre_x=centre_x, 
            centre_y=centre_y, 
            inclination=inclination,
            axis_ratio=axis_ratio,
            
            intensity=intensity,
            truncation_radius = truncation_radius,
            truncation_strength=truncation_strength,
            powerlaw_slope=powerlaw_slope,
            boxness=boxness
      )      
      
      
class King(Ellip, Light_Profile):
  def __init__(
  self, 
  centre_x: Centre_x, 
  centre_y: Centre_y, 
  inclination: Inclination,
  axis_ratio: Axis_ratio,
  
  
  intensity: Intensity,
  core_radius: Core_radius,
  truncation_radius: Truncation_radius,
  powerlaw_index: Powerlaw_index,
  
  boxness: Boxness

):
    super().__init__(centre_x=centre_x, centre_y=centre_y, inclination=inclination, axis_ratio=axis_ratio, boxness=boxness)
    self.intensity = intensity
    self.core_radius = core_radius
    self.truncation_radius = truncation_radius
    self.powerlaw_index = powerlaw_index


  def image_2d_from_(self, radius, mode="updating_value"):
    partA = torch.pow((1+torch.pow((self.truncation_radius.updating_v(mode)/self.core_radius.updating_v(mode)), 2)), 1/self.powerlaw_index.updating_v(mode))
    
    partB = torch.pow((1+torch.pow((radius/self.core_radius.updating_v(mode)), 2)), 1/self.powerlaw_index.updating_v(mode))
    
    return self.intensity.updating_v(mode) * torch.pow(1 - (1/partA), -self.powerlaw_index.updating_v(mode)) * torch.pow(1/partB - 1/partA, self.powerlaw_index.updating_v(mode))
                    
  
  def image_2d_grid_from_(self, grid, mode="updating_value"):
    return self.image_2d_from_(self.make_radius(grid, mode), mode)
  

  def param(self, name: str) -> dict: 
    flex_dic = {}
    fix_dic = {}
    test_dic = {}
    
    for param in [self.centre_x, self.centre_y, self.inclination, self.axis_ratio, self.intensity, self.core_radius, self.truncation_radius, self.powerlaw_index, self.boxness]:
      if id(param) not in test_dic:
        value = name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = value
        if param.if_fit == True:
          flex_dic[value] = param
        else:
          fix_dic[value] = param
                
      else:
        _value = test_dic[id(param)]
        _new_value = _value + "  and  " + name + "_"+self.__class__.__name__ + "_" +param.__class__.__name__
        test_dic[id(param)] = _new_value
    
        if param.if_fit == True:
          flex_dic[_new_value] = flex_dic.pop(_value)
        else:
          fix_dic[_new_value] = flex_dic.pop(_value)
    return flex_dic, fix_dic, test_dic

      
  @classmethod
  def definition(
    cls,
    centre_x: Centre_x, 
    centre_y: Centre_y, 
    inclination: Inclination,
    axis_ratio: Axis_ratio,
    
    intensity: Intensity,
    core_radius: Core_radius,
    truncation_radius: Truncation_radius,
    powerlaw_index: Powerlaw_index,
    
    boxness: Boxness = None,
  ) -> "King":
      """
      """
      length = []
      args, _, _, values = inspect.getargvalues(inspect.currentframe())
      if boxness == None:
        boxness = Boxness(parameters=torch.zeros(length[0]))

      for _, arg in enumerate(args[1: ]):
        length.append(values[arg].param_length)
      assert all(x == length[0] for x in length), "the length of parameters are not the same!"
      
      return King(
            centre_x=centre_x, 
            centre_y=centre_y, 
            inclination=inclination,
            axis_ratio=axis_ratio,
            
            intensity=intensity,
            core_radius = core_radius,
            truncation_radius=truncation_radius,
            powerlaw_index=powerlaw_index,
            boxness=boxness
      )      
      
    
    
# def image_2d_from_(self, radius, output=False):
  #   if output==False:
  #     print("aaa")
  #     l_tot1 = torch.pow(self.effective_radius.value_b, 2) * 2 * torch.pi * self.sersic_index.value_b * torch.exp(self.sersic_constant()) * self.axis_ratio.value_b
  #     gam = torch.exp(torch.lgamma(2 * self.sersic_index.value_b))
  #     l_tot2_log = torch.log10(gam) - 2 * self.sersic_index.value_b * torch.log10(self.sersic_constant())
  #     l_tot = l_tot1 * torch.pow(10, l_tot2_log)
  #     F_tot = torch.pow(10, (-0.4 * self.magnitude.value_b)) * self.R_box 
  #     return (F_tot/l_tot) * torch.exp(-self.sersic_constant()
  #                           * ( torch.pow(radius / self.effective_radius.value_b, (1.0 / self.sersic_index.value_b )) - 1))
  #   else:
  #     with torch.no_grad():
  #       l_tot1 = torch.pow(self.effective_radius.b_bV, 2) * 2 * torch.pi * self.sersic_index.b_bV * torch.exp(self.sersic_constant(output=True)) * self.axis_ratio.b_bV
      
  #     gam = torch.exp(torch.lgamma(2 * self.sersic_index.b_bV))
  #     l_tot2_log = torch.log10(gam) - 2 * self.sersic_index.b_bV * torch.log10(self.sersic_constant(output=True))
  #     l_tot = l_tot1 * torch.pow(10, l_tot2_log)
  #     F_tot = torch.pow(10, (-0.4 * self.magnitude.b_bV)) * self.R_box 

  #     with torch.no_grad():
  #       result = (F_tot/l_tot) * torch.exp(-self.sersic_constant(output=True)
  #                           * ( torch.pow(radius / self.effective_radius.b_bV, (1.0 / self.sersic_index.b_bV )) - 1)) 
  #     return result