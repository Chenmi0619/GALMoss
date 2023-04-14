from collections.abc import Sequence as Seq
from typing import Union, Tuple
import numpy as np
import torch
from functools import wraps
from typing import List
from parameters import Centre_x, Centre_y, Inclination, Axis_ratio
import math 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Light_Profile:
  pass

def revise_grid(func):
    """
    To check if the center turns zero

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
        grid,
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

        x_c, y_c = cls.unsqueeze_center
        
        # only use on test

        x = grid[0].to(device) - x_c
        y = grid[1].to(device) - y_c
        
        # x = grid[0] - x_c
        # y = grid[1] - y_c
        grid = [x, y]
        return func(cls, grid, *args, **kwargs)

    return wrapper
 
  
def check_center(func):
    """
    To check if the center turns zero

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
        grid,
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
        
        radius = func(cls, grid, *args, **kwargs)
        with np.errstate(all="ignore"):  # Division by zero fixed via isnan
            radius_index = torch.where(
                radius < 0.001, 0.001 / radius, 1.0
            )
            radius = radius * radius_index[:, None]
        radius[torch.isnan(radius)] = 0.001
        return radius
    return wrapper
  
class Ellip(Light_Profile):
  """
  Base class for elliptical profile

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
    centre_x: Centre_x, 
    centre_y: Centre_y, 
    inclination: Inclination,
    axis_ratio: Axis_ratio,
  ):
    self.centre_x = centre_x
    self.centre_y = centre_y
    self.inclination = inclination
    self.axis_ratio = axis_ratio
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
  def inclination_second(self):
    return (self.inclination.update_value) * torch.pi / 180

  @property
  def unsqueeze_center(self):
    x_x = self.centre_x.update_value
    x_y = self.centre_y.update_value
    x_c = torch.unsqueeze(torch.unsqueeze(x_x, dim=1), dim=1)
    y_c = torch.unsqueeze(torch.unsqueeze(x_y, dim=1), dim=1)
    
    return x_c, y_c

  @property
  def unsqueeze_inclination_second(self):
    return torch.unsqueeze(torch.unsqueeze(self.inclination_second, dim=1), dim=1)

  @property
  def unsqueeze_axis_ratio(self):
    return torch.unsqueeze(torch.unsqueeze(self.axis_ratio.update_value, dim=1), dim=1)

  #@check_center
  @revise_grid
  def make_radius(self, grid: Tuple[torch.Tensor, torch.Tensor]):
    self.x = grid[0]
    self.y = grid[1]
    
    self.x_maj = torch.cos(self.unsqueeze_inclination_second) * self.x + torch.sin(self.unsqueeze_inclination_second) * self.y
    self.x_min = -self.x * torch.sin(self.unsqueeze_inclination_second) + self.y * torch.cos(self.unsqueeze_inclination_second)
    self.radius = torch.pow(self.x_maj ** 2 + (self.x_min / self.unsqueeze_axis_ratio) ** 2, 0.5)
    return self.radius
  
# if __name__ == "__main__":
#   data_shape = [10, 10]
#   x = torch.Tensor(list(range(1, (data_shape[1])+1)))
#   y = torch.flip(torch.Tensor(list(range(1, (data_shape[0])+1))), [0])
#   xy = torch.meshgrid(y, x)
  
#   center = Centre(parameters=[5, 5])
#   incli = Inclination(parameters=100)
#   axi = Axis_ratio(parameters=0.2)
#   test = Ellip(centre= center, inclination=incli, axis_ratio= axi)
#   aa = test.make_radius(grid=xy)
  
#   print(aa)

