from typing import Tuple
import numpy as np
import torch
from functools import wraps
from galmoss.Parameter.parameters import Centre_x, Centre_y, Inclination, Axis_ratio, Boxness
device = 'cuda' if torch.cuda.is_available() else 'cpu'



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
        mode,
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
        
        y = grid[0] -( grid[0].shape[0] - cls.centre_y.updating_v(mode)) + 1e-6
        x = grid[1] - cls.centre_x.updating_v(mode) + 1e-6
        grid = [x, y]         
        return func(cls, grid, mode, *args, **kwargs)

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
  
class Ellip():
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
    boxness: Boxness
  ):
    self.centre_x = centre_x
    self.centre_y = centre_y
    self.inclination = inclination
    self.axis_ratio = axis_ratio
    self.boxness = boxness
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
  def R_box(self, mode):
      beta = torch.exp(torch.lgamma(1/(2 + self.boxness.updating_v(mode)))) * torch.exp(torch.lgamma(1/(2 + self.boxness.updating_v(mode)))) / torch.exp(torch.lgamma(2/(2 + self.boxness.updating_v(mode))))
      return torch.pi * (2 + self.boxness.updating_v(mode)) / (2 * beta)

  # @property  
  # def inclination_second(self):
  #   return (self.inclination.update_value) * torch.pi / 180

  # @property
  # def unsqueeze_center(self):
  #   x_x = self.centre_x.update_value
  #   x_y = self.centre_y.update_value
  #   x_c = torch.unsqueeze(torch.unsqueeze(x_x, dim=1), dim=1)
  #   y_c = torch.unsqueeze(torch.unsqueeze(x_y, dim=1), dim=1)
  #   return x_c, y_c

  # @property
  # def unsqueeze_inclination_second(self):
  #   return torch.unsqueeze(torch.unsqueeze(self.inclination_second, dim=1), dim=1)


  #@check_center
  @revise_grid
  def make_radius(self, grid: Tuple[torch.Tensor, torch.Tensor], mode="updating_value"):
    self.y = grid[0]
    self.x = grid[1] 

    self.x_maj = torch.abs(torch.cos(self.inclination.updating_v(mode)) * self.x + torch.sin(self.inclination.updating_v(mode)) * self.y)
    self.x_min = torch.abs(-self.x * torch.sin(self.inclination.updating_v(mode)) + self.y * torch.cos(self.inclination.updating_v(mode)))
    self.radius = torch.pow(torch.pow(self.x_maj, (self.boxness.updating_v(mode) + 2)) + torch.pow((self.x_min/ self.axis_ratio.updating_v(mode)), (self.boxness.updating_v(mode) + 2)), 1/ (self.boxness.updating_v(mode) + 2))

    return self.radius
  

