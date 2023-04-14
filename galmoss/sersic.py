from ellip import Ellip
from Parameters import AbstractParameters
from parameters import Centre_x, Centre_y, Inclination, Axis_ratio, Effective_radius, Intensity, Sersic_index
# from fitting import Fitting
from collections.abc import Sequence as Seq
from typing import Union, Tuple
import numpy as np
import torch
from functools import wraps
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 需要补一个画图模块

    
  
class Sersic(Ellip):
  def __init__(
  self, 
  centre_x: Centre_x, 
  centre_y: Centre_y, 
  inclination: Inclination,
  axis_ratio: Axis_ratio,
  
  effective_radius: Effective_radius,
  intensity: Intensity,
  sersic_index: Sersic_index

):
    super().__init__(centre_x=centre_x, centre_y=centre_y, inclination=inclination, axis_ratio=axis_ratio)
    self.effective_radius = effective_radius
    self.intensity = intensity
    self.sersic_index = sersic_index
   
  @property 
  def sersic_constant(self) -> float:
    """
    A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
    total integrated light.
    """
    return (
        (2 * self.sersic_index.update_value)
        - (1.0 / 3.0)
        + (4.0 / (405.0 * self.sersic_index.update_value))
        + (46.0 / (25515.0 * self.sersic_index.update_value**2))
        + (131.0 / (1148175.0 * self.sersic_index.update_value**3))
        - (2194697.0 / (30690717750.0 * self.sersic_index.update_value**4))
    )
    
# def sersic_constant(a):
# """
# A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
# total integrated light.
# """
#   return ( (2 * a)- (1.0 / 3.0)+ (4.0 / (405.0 * a))+ (46.0 / (25515.0 * a)) + (131.0 / (1148175.0 * a))- (2194697.0 / (30690717750.0 * a)))
#     (2)- ((4.0) / (405.0 * torch.pow(a, 2)))- ((46.0 * 2) / (25515.0 * torch.pow(a, 3)))- ((131.0 * 3) / (1148175.0 * torch.pow(a, 4))) + ((2194697.0 * 4) / (30690717750.0 * torch.pow(a, 5)))
    
  @property 
  def sersic_constant_gradient(self) -> float:
    """
    A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
    total integrated light.
    """
    _scale = (self.sersic_index.param_scale[:, 1] - self.sersic_index.param_scale[:, 0])/2
    return (
        (2 * _scale)
        - ((4.0 * _scale) / (405.0 * torch.pow(self.sersic_index.update_value, 2)))
        - ((46.0 * 2 * _scale) / (25515.0 * torch.pow(self.sersic_index.update_value, 3)))
        - ((131.0 * 3 * _scale) / (1148175.0 * torch.pow(self.sersic_index.update_value, 4)))
        + ((2194697.0 * 4 * _scale) / (30690717750.0 * torch.pow(self.sersic_index.update_value, 5)))
    )
    
    
  def image_2d_from_(self, radius):
    re_ = torch.unsqueeze(torch.unsqueeze(self.effective_radius.update_value, dim=1), dim=1)
    q_ = torch.unsqueeze(torch.unsqueeze(self.axis_ratio.update_value, dim=1), dim=1)
    m_tot = torch.unsqueeze(torch.unsqueeze(self.intensity.update_value, dim=1), dim=1)
    
    bn = torch.unsqueeze(torch.unsqueeze(self.sersic_constant, dim=1), dim=1)
    n_ = torch.unsqueeze(torch.unsqueeze(self.sersic_index.update_value, dim=1), dim=1)
    l_tot1 = torch.pow(re_, 2) * 2 * np.pi * n_ * torch.exp(bn) * q_ 
    gam = torch.exp(torch.lgamma(2 * n_))
    l_tot2_log = torch.log10(gam) - 2 * n_ * torch.log10(bn)
    l_tot = l_tot1 * torch.pow(10, l_tot2_log)
    # mag_tot = (-2.5 * torch.log10(l_tot))
    F_tot = torch.pow(10, (-0.4 * m_tot)) # * 12599


    
    
                        
    return (F_tot/l_tot) * torch.exp(-bn
                          * ( torch.pow(radius / re_, (1.0 / n_ )) - 1))
  
  
  def image_2d_grid_from_(self, grid):
    return self.image_2d_from_(self.make_radius(grid))
  
  
  # def image_uncer__2d_grid_from_(self, grid):
  #   print("grid", np.shape(grid))
  #   print(grid)
  #   # print("aaaa", np.shape(self.image_2d_from_(self.make_radius(grid))))
  #   print("aaaa", np.shape(self.make_radius(grid)))
  #   print(torch.sum(self.image_2d_from_(self.make_radius(grid)), dim=0).view((1, -1)))
  #   return torch.sum(self.image_2d_from_(self.make_radius(grid)), dim=0).view((1, -1))
  # # @property
  # def data(self):
  #   flex_dic = {}
  #   fix_dic = {}
    
  #   for param in [self.centre, self.inclination, self.axis_ratio, self.axis_ratio,
  #                 self.effective_radius, self.intensity, self.sersic_index]:
  #     if param.if_fit == True:
  #       flex_dic[self.dict()] = param
  #     else:
  #       fix_dic[self.dict()] = param
    
    # return flex_dic, fix_dic
  

  def param(self, name: str) -> dict: 
    flex_dic = {}
    fix_dic = {}
    test_dic = {}
    
    for param in [self.centre_x, self.centre_y, self.inclination, self.axis_ratio, self.axis_ratio,
                  self.effective_radius, self.intensity, self.sersic_index]:
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
  
  
  
  def uncertainty(self, loss2, aaa):
      a = torch.Tensor([]).to(device)
      print(np.shape(a))
      for _, value in self.__dict__.items():
        
          if isinstance(value, AbstractParameters):
            print(_)
            bbb = value.cal_uncertainty(loss2, aaa)
            print(np.shape(a), np.shape(bbb))
            a = torch.cat((a, bbb), dim=0)
            
            # print(bbb)
            
      a = a.view(len(self.__dict__.items()), -1)      
      print(np.shape(a)) 
      k = np.array(a)   
      c = torch.inverse(a)
      print("a", c)
      
  
  def grad_from(self, grid, psf, index):
    return Grad_Sersic(self.centre_x, self.centre_y, self.inclination, 
                       self.axis_ratio, self.effective_radius,
                       self.intensity, self.sersic_index, grid, 
                       self.image_2d_grid_from_(grid), self.make_radius(grid), psf, index)

class Abstract_Grad_ellip:
  def __init__(
  self, 
  centre_x: Centre_x, 
  centre_y: Centre_y, 
  inclination: Inclination,
  axis_ratio: Axis_ratio,
  grid
):  
    self.centre_x = centre_x
    self.centre_y = centre_y
    self.inclination = inclination
    self.axis_ratio = axis_ratio
    self.grid = grid
  
  @property
  def inclination_second(self):
    return (self.inclination.sq_V) * torch.pi / 180
  
  @property
  def x_grid(self):
    return self.grid[0] - self.centre_x.sq_V
  
  @property
  def y_grid(self):
    return self.grid[1] - self.centre_y.sq_V
    
  @property
  def x_maj(self):
    return torch.cos(self.inclination_second) * self.x_grid + torch.sin(self.inclination_second) * self.y_grid
  
  @property
  def x_min(self):
    return -torch.sin(self.inclination_second) * self.x_grid + torch.cos(self.inclination_second) * self.y_grid

  @property  
  def p_x_maj_p_inclination(self):
    return -torch.sin(self.inclination_second) * self.x_grid * (self.inclination.grad_C * np.pi / 
                                                                180) + torch.cos(self.inclination_second) * self.y_grid * (self.inclination.grad_C * np.pi / 180)
    
  @property  
  def p_x_min_p_inclination(self):
    return -torch.cos(self.inclination_second) * self.x_grid * (self.inclination.grad_C * np.pi / 
                                                                180) - torch.sin(self.inclination_second) * self.y_grid * (self.inclination.grad_C * np.pi / 180)

    
    
  @property  
  def p_radial_p_axis(self):
    return 0.5 * torch.pow(self.x_maj**2 + 
                           (self.x_min / self.axis_ratio.sq_V)**2, -0.5) * ((-self.x_min**2 * 2 * self.axis_ratio.grad_C) / (self.axis_ratio.sq_V ** 3))
  
  @property  
  def p_radial_p_inclination(self):
    return 0.5 * torch.pow(self.x_maj**2 + 
                           (self.x_min / self.axis_ratio.sq_V)**2, -0.5) * (2 * self.x_maj * self.p_x_maj_p_inclination + 
                                                                       (2 * self.x_min * self.p_x_min_p_inclination) / self.axis_ratio.sq_V**2)

  
  @property  
  def p_radial_p_center_x(self):

    return 0.5 * torch.pow(self.x_maj**2 +
                           (self.x_min / 
                            self.axis_ratio.sq_V)**2, -0.5) * (2 * self.x_maj * torch.cos(self.inclination_second) * (-self.centre_x.grad_C) 
                                                                            + (2 * self.x_min * -torch.sin(self.inclination_second) * (-self.centre_x.grad_C)) / 
                                                                            self.axis_ratio.sq_V**2)
  @property  
  def p_radial_p_center_y(self):
    
    return 0.5 * torch.pow(self.x_maj**2 +
                           (self.x_min / 
                            self.axis_ratio.sq_V)**2, -0.5) * (2 * self.x_maj * torch.sin(self.inclination_second) * (-self.centre_x.grad_C) 
                                                                            + (2 * self.x_min * torch.cos(self.inclination_second) * (-self.centre_y.grad_C)) / 
                                                                            self.axis_ratio.sq_V**2)
    
  
class Abstract_Grad_Sersic(Abstract_Grad_ellip):
  def __init__(
  self, 
  centre_x: Centre_x, 
  centre_y: Centre_y, 
  inclination: Inclination,
  axis_ratio: Axis_ratio,
  
  effective_radius: Effective_radius,
  intensity: Intensity,
  sersic_index: Sersic_index,
  
  grid,
  model,
  radius

):
    super().__init__(
    centre_x=centre_x, 
    centre_y=centre_y, 
    inclination=inclination, 
    axis_ratio=axis_ratio, 
    grid=grid)
    
    self.effective_radius = effective_radius
    self.intensity = intensity
    self.sersic_index = sersic_index
    
    self.model = model
    self.radius = radius
  @property 
  def sersic_constant(self) -> float:
    """
    A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
    total integrated light.
    """
    return (
        (2 * self.sersic_index.update_value)
        - (1.0 / 3.0)
        + (4.0 / (405.0 * self.sersic_index.update_value))
        + (46.0 / (25515.0 * self.sersic_index.update_value**2))
        + (131.0 / (1148175.0 * self.sersic_index.update_value**3))
        - (2194697.0 / (30690717750.0 * self.sersic_index.update_value**4))
    )
    
  @property
  def sersic_constant_sq_V(self):
    return torch.unsqueeze(torch.unsqueeze(self.sersic_constant, dim=1), dim=1)
    
  @property 
  def sersic_constant_gradient(self) -> float:
    """
    A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
    total integrated light.
    """
    return (
        (2)
        - ((4.0) / (405.0 * torch.pow(self.sersic_index.update_value, 2)))
        - ((46.0 * 2) / (25515.0 * torch.pow(self.sersic_index.update_value, 3)))
        - ((131.0 * 3) / (1148175.0 * torch.pow(self.sersic_index.update_value, 4)))
        + ((2194697.0 * 4) / (30690717750.0 * torch.pow(self.sersic_index.update_value, 5)))
    )

  @property
  def sersic_constant_gradient_sq_V(self):
    return torch.unsqueeze(torch.unsqueeze(self.sersic_constant_gradient, dim=1), dim=1)
    
  def grad_intensity(self):   
    return (self.model * torch.log(torch.Tensor([10])).to(device)* -0.4 * self.intensity.grad_C)
  
  def grad_sersic_index(self):
    A =  torch.pow(10, (-0.4 * self.intensity.sq_V)) / (
      torch.pow(self.effective_radius.sq_V, 2) * 2 * np.pi * self.axis_ratio.sq_V) / (
        torch.pow(10, torch.log10(torch.exp(torch.lgamma(2 * self.sersic_index.sq_V))) - 
                  2 * self.sersic_index.sq_V * torch.log10(self.sersic_constant_sq_V))  * 
        self.sersic_index.sq_V * torch.exp(self.sersic_constant_sq_V)) ** 2
    print("A2", A)
    # B = (torch.pow(10, torch.log10(torch.exp(torch.lgamma(2 * self.sersic_index.sq_V))) - 
    #                2 * self.sersic_index.sq_V * torch.log10(self.sersic_constant_sq_V))  * 
    #      self.sersic_index.sq_V * torch.exp(self.sersic_constant_sq_V)) ** 2
    
    grad_C = torch.exp(-self.sersic_constant_sq_V * 
                    (torch.pow(self.radius / self.effective_radius.sq_V, 
                               (1.0 / self.sersic_index.sq_V)) - 1)) * ( 
                              - (self.sersic_index.grad_C * self.sersic_constant_gradient_sq_V * (
                                torch.pow(self.radius / self.effective_radius.sq_V, 
                                          (1.0 / self.sersic_index.sq_V)) - 1)  + 
                                 self.sersic_constant_sq_V * (torch.log(self.radius / self.effective_radius.sq_V) * 
                                                              torch.pow(self.radius / self.effective_radius.sq_V, 
                                  (1.0 / self.sersic_index.sq_V)) * (-1 / torch.pow(self.sersic_index.sq_V, 2)) * self.sersic_index.grad_C)))
    C = torch.exp(-self.sersic_constant_sq_V * (torch.pow(self.radius / self.effective_radius.sq_V, (1.0 / self.sersic_index.sq_V)) - 1))
    
    grad_D = (torch.exp(torch.lgamma(2 *  (self.sersic_index.sq_V+ 1e-4)))  - 
              torch.exp(torch.lgamma(2 * self.sersic_index.sq_V))) * 4.95 / 1e-4 * (
                torch.pow(10, - 2 * self.sersic_index.sq_V * torch.log10(self.sersic_constant_sq_V))  
                * self.sersic_index.sq_V * torch.exp(self.sersic_constant_sq_V)) + torch.exp(
                  torch.lgamma(2 * self.sersic_index.sq_V)) * (torch.pow(10, - 2 * self.sersic_index.sq_V * 
                                                                         torch.log10(self.sersic_constant_sq_V))   
                    * (-2 * ( torch.log(self.sersic_constant_sq_V) * self.sersic_index.grad_C + 
                             self.sersic_index.sq_V * (self.sersic_index.grad_C * self.sersic_constant_gradient_sq_V / 
                                                       self.sersic_constant_sq_V))) * self.sersic_index.sq_V * 
                    torch.exp(self.sersic_constant_sq_V) + torch.pow(10, - 2 * self.sersic_index.sq_V * torch.log10(self.sersic_constant_sq_V))  
                    * (torch.exp(self.sersic_constant_sq_V) * self.sersic_index.grad_C + (self.sersic_index.sq_V*torch.exp(self.sersic_constant_sq_V) 
                                                                      * self.sersic_index.grad_C * self.sersic_constant_gradient_sq_V)))                           
    D = torch.pow(10, torch.log10(torch.exp(torch.lgamma(2 * self.sersic_index.sq_V))) - 2 * self.sersic_index.sq_V * 
                  torch.log10(self.sersic_constant_sq_V)) * self.sersic_index.sq_V * torch.exp(self.sersic_constant_sq_V) 
  
    return (A) * (grad_C * D - C * grad_D)
  
  def grad_effective_radius(self):
      A = torch.pow(10, (-0.4 * self.intensity.sq_V)) / (torch.pow(
        10, torch.log10(torch.exp(torch.lgamma(2 * self.sersic_index.sq_V))) 
         - 2 * self.sersic_index.sq_V * torch.log10(self.sersic_constant_sq_V)) * 
          self.sersic_index.sq_V * 2 * np.pi * self.axis_ratio.sq_V * torch.exp(self.sersic_constant_sq_V) * 
          torch.pow(self.effective_radius.sq_V, 4))
      
      grad_B = torch.exp(-self.sersic_constant_sq_V * (torch.pow(self.radius / self.effective_radius.sq_V, 
                        (1.0 / self.sersic_index.sq_V)) - 1)) * (-self.sersic_constant_sq_V) * (
                         (1/self.sersic_index.sq_V) * torch.pow((self.radius / self.effective_radius.sq_V), 
                        (1 / self.sersic_index.sq_V -1)) * ((- self.radius * self.effective_radius.grad_C)/self.effective_radius.sq_V**2)) 
      B = torch.exp(-self.sersic_constant_sq_V * (torch.pow(self.radius / self.effective_radius.sq_V, (1.0 / self.sersic_index.sq_V)) - 1))
      grad_C = 2 * self.effective_radius.sq_V * self.effective_radius.grad_C
      C = self.effective_radius.sq_V ** 2
      return A * (grad_B * C - B* grad_C)
    
  def grad_axis_ratio(self): 
    A = torch.pow(10, (-0.4 * self.intensity.sq_V)) / (
      torch.pow(10, torch.log10(torch.exp(torch.lgamma(2 * self.sersic_index.sq_V))) - 
                2 * self.sersic_index.sq_V * torch.log10(self.sersic_constant_sq_V))  * 
                self.sersic_index.sq_V * 2 * np.pi * torch.exp(self.sersic_constant_sq_V) * 
                torch.pow(self.effective_radius.sq_V, 2) * torch.pow(self.axis_ratio.sq_V, 2))
    grad_B = torch.exp(-self.sersic_constant_sq_V * (torch.pow(self.radius / self.effective_radius.sq_V, (
              1.0 / self.sersic_index.sq_V)) - 1)) * (-self.sersic_constant_sq_V) * ((1/self.sersic_index.sq_V) * torch.pow(
              self.radius/self.effective_radius.sq_V, 1/self.sersic_index.sq_V -1) * (1/self.effective_radius.sq_V) * self.p_radial_p_axis)
    B = torch.exp(-self.sersic_constant_sq_V * (torch.pow(self.radius / self.effective_radius.sq_V, (
      1.0 / self.sersic_index.sq_V)) - 1))
    grad_C = self.axis_ratio.grad_C
    C = self.axis_ratio.sq_V
    return A * (grad_B * C - B * grad_C)
  
  def grad_inclination(self):

    return self.model* (-self.sersic_constant_sq_V) * ((1/self.sersic_index.sq_V) * 
               torch.pow(self.radius/self.effective_radius.sq_V, 1/self.sersic_index.sq_V -1) * (
                 1/self.effective_radius.sq_V) * self.p_radial_p_inclination)
    
  def grad_center_x(self):
    return self.model* (-self.sersic_constant_sq_V) * ((1/self.sersic_index.sq_V) * 
           torch.pow(self.radius/self.effective_radius.sq_V, 1/self.sersic_index.sq_V -1) * 
           (1/self.effective_radius.sq_V) * self.p_radial_p_center_x)
    
  def grad_center_y(self):
    return self.model* (-self.sersic_constant_sq_V) * ((1/self.sersic_index.sq_V) * 
           torch.pow(self.radius/self.effective_radius.sq_V, 1/self.sersic_index.sq_V -1) * 
           (1/self.effective_radius.sq_V) * self.p_radial_p_center_y) 


class Grad_Sersic(Abstract_Grad_Sersic):
  def __init__(
    self, 
    centre_x: Centre_x,
    centre_y: Centre_y,
    inclination: Inclination,
    axis_ratio: Axis_ratio,
    effective_radius: Effective_radius,
    intensity: Intensity,
    sersic_index: Sersic_index,
    grid: tuple[torch.Tensor, torch.Tensor],
    model: torch.Tensor,
    radius: torch.Tensor,
    psf_data: torch.Tensor,
    galaxy_trian_index: int
    ): 
    self.grid = grid
    self.model = model
    self.radius = radius
    self.psf_data = psf_data
    self.galaxy_train_index = galaxy_trian_index
    
    
    self.centre_x=centre_x 
    self.centre_y=centre_y
    self.inclination=inclination
    self.axis_ratio=axis_ratio
    
    self.effective_radius = effective_radius
    self.intensity = intensity
    self.sersic_index = sersic_index
                     
    
    super().__init__(centre_x=centre_x, 
                     centre_y=centre_y, 
                     inclination=inclination, 
                     axis_ratio=axis_ratio, 
                     
                     effective_radius = effective_radius,
                     intensity = intensity,
                     sersic_index = sersic_index,
                     
                     grid=grid,
                     model=model,
                     radius=radius)
    
  def conv(self, data):
    normalization = torch.unsqueeze(torch.unsqueeze(torch.sum(self.psf_data, dim=[1, 2]), 1), 1)
    data = data.float()
    data_after_psf = torch.squeeze(F.conv2d(data, self.psf_data.unsqueeze(1),
                                padding="same", groups=len(self.galaxy_train_index)))
    return data_after_psf/normalization
  
  def Conv_center_x(self):
    return self.conv(self.grad_center_x())
  
  def Conv_center_y(self):
    return self.conv(self.grad_center_y())
  
  def Conv_axis_ratio(self):
    return self.conv(self.grad_axis_ratio())    
  def Conv_inclination(self):
    return self.conv(self.grad_inclination())    
  def Conv_intenisty(self):
    return self.conv(self.grad_intensity())      
  def Conv_effective_radius(self):
    return self.conv(self.grad_effective_radius())    
    
  def Conv_sersic_index(self):
    return self.conv(self.grad_sersic_index())
  
  def grad_backward(self):
    self.centre_x.grad_store(self.Conv_center_x())
    self.centre_y.grad_store(self.Conv_center_y())
    self.axis_ratio.grad_store(self.Conv_axis_ratio())
    self.inclination.grad_store(self.Conv_inclination())
    self.intensity.grad_store(self.Conv_intenisty())
    self.effective_radius.grad_store(self.Conv_effective_radius())
    self.sersic_index.grad_store(self.Conv_sersic_index())
    
    
    
    
# class Grad:
#   def __init__(self, fit: Fitting): 
#     self.fit = Fitting
    
    
           
    
class Profiles(): 
  def __init__(self, **kwargs):
    self.flex = {}
    self.fix = {}
    self.test = {}
    for name, val in kwargs.items():
      flexx, fixx, test = val.param(name=name)
      print("type", flexx, type(flexx.values))
      self.flex.update(flexx)
      self.fix.update(fixx)
      self.test.update(test)
      setattr(self, name, val)  


# if __name__ == "__main__":
#   data_shape = [10, 10]
#   x = torch.Tensor(list(range(1, (data_shape[1])+1)))
#   y = torch.flip(torch.Tensor(list(range(1, (data_shape[0])+1))), [0])
#   xy = torch.meshgrid(y, x)
  
#   center_x = Centre_x(parameters=center_yy,
#                       scale=[5, 5])
#   center_y = Centre_y(parameters=center_yy,
#                       scale=[50, 70])
#   center = Centre(parameters=[5, 5])
#   incli = Inclination(parameters=100)
#   axi = Axis_ratio(parameters=0.2)
#   effr = Effective_radius(parameters=5)
#   index = Sersic_index(parameters=3)
#   inten = Intensity(parameters=16)
  
#   test = Sersic(centre= center, inclination=incli, axis_ratio= axi, effective_radius=effr,
#                 sersic_index=index, intensity=inten)
#   test222 = Sersic(centre= center, inclination=incli, axis_ratio= axi, effective_radius=effr,
#                 sersic_index=index, intensity=inten)
#   # aa = test.image_2d_grid_from_(grid=xy)
#   flex22, fix22 = test222.param(name="test222")
#   flex, fix = test.param(name="test")
#   a = {}
#   a.update(flex22)
#   a.update(flex)
#   print(a)
#   # print(bb.value())
