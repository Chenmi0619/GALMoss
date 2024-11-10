from typing import Tuple, Union
import numpy as np
import torch
from functools import wraps

from galmoss.Parameter.basic import Parameters


def revise_grid(func):
    """
    Transform the grid to galaxy-centered, and then add a shift to avoid 
    the values near to zero. Grid values which are near to zero may cause 
    NaN or Inf in calculations.

    Parameters
    ----------
    func : (profile, *args, **kwargs) -> torch.tensor
        A function that receives a original grid.

    Returns
    -------
        A function that can accept grids.
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
        cls
            The class that owns the function.
        grid
            The coordinates in the original domain.
        mode:
            The value mode, which are used to call 
            various mode of parameter values.

        Returns
        -------
            The grid being transformed and revise from zero.
        """
        y = grid[0] - (grid[0].shape[0] - cls.cen_y.value(mode)) + 1e-6
        x = grid[1] - cls.cen_x.value(mode) + 1e-6

        grid = [x, y]         
        return func(cls, grid, mode, *args, **kwargs)

    return wrapper


class LightProfile:
    
    def __init__(self): 
        """
        Abstract base class for an light-profile.

        Each light profile has an equation, which represents the distribution 
        of surface brightness. The model image can be calculated given an 
        2D grid.
        
        Attribute
        ---------
        psf
            To control whether this profile will be convolved with PSF 
            imagec later.
        saving
            To control whether the model image of this profile will be
            saved in image block in the end of fitting process.
        """           
        self.psf = True
        self.saving = True
        
    def image_via_grid_from(
        self, 
        grid: Union[torch.tensor, torch.tensor],
        mode: str):
            """
            Returns the light profile's image calculated from a 2D grid.

            Parameters
            ----------
            grid
                The galaxy-centered grid.
            mode
                Determine which kind of values will be called in 
                parameter object.

            Returns
            -------
            image
                The model image calculated with the grid.
            """
            raise NotImplementedError()    

    def manage_params(self, name: str) -> dict:
        """
        To manage the parameters in this profile. The parameters will be 
        divide into fixed/variable params based on the value of their 
        attribute `fit`, and be stored in correspoding dictionaries. These
        dictionaries can be called by `variable_dic` and `fixed_dic`. This 
        function will also check the length of these parameters, which 
        will raise error if they are not concise.

        Parameters
        ----------
        name
            The user-defined name of this profile, which will be used in name 
            the keys of each values in dictionaries like "profile_name" + "_"
            + "param_name" 

        """        
        variable_dic = {}
        fixed_dic = {}
        param_length = []
        for param_name in dir(self):
            param = getattr(self, param_name)            
            if isinstance(param, Parameters):
                param.param_name = param_name
                param_length.append(param.param_length)
                idx = name + "_" + param_name
                if param.fit:
                    variable_dic[idx] = param 
                else:
                    fixed_dic[idx] = param
        assert all(x == param_length[0] 
                   for x in param_length), ("the length of parameters"
                                            "are not the same!")
        self.variable_dic = variable_dic
        self.fixed_dic = fixed_dic


class Ellip(LightProfile):
     
    def __init__(
        self, 
        cen_x: Parameters, 
        cen_y: Parameters, 
        pa: Parameters,
        axis_r: Parameters,
        box: Parameters
    ):
        """
        Abstract base class for an radial profile.

        Each radial profile has a transition from grid to radial. 
        
        Parameters
        ---------
        cen_x
            The galaxy center in x-xis in arc-second.
        cen_y
            The galaxy center in y-xis in arc-second.
        pa
            The position angle of the ellipse.
        axis_r
            The axis ratio of the ellipse.
        box
            The boxness of the ellipse.    
        """        
        super().__init__()
        self.cen_x = cen_x
        self.cen_y = cen_y
        self.pa = pa
        self.axis_r = axis_r    
        
        if box == None:
            self.box = Parameters(parameters=torch.zeros(cen_x.param_length), 
                                  fit=False)   
        else:
            self.box = box  
        
    def R_box(self, mode):
        """
        The geometric correction factor in Ie to account for deviations 
        from a perfect ellipse, influenced by the level of diskiness or 
        boxiness. Specifically, when box = 0, indicating a perfect 
        ellipse, R_box = 1, implying no geometric correction. 
        """         
        beta = (torch.exp(torch.lgamma(1/(2 + self.box.value(mode)))) 
                * torch.exp(torch.lgamma(1/(2 + self.box.value(mode)))) 
                / torch.exp(torch.lgamma(2/(2 + self.box.value(mode)))))
        
        return torch.pi * (2 + self.box.value(mode)) / (2 * beta)

    @revise_grid
    def make_radius(self, 
                    grid: Tuple[torch.Tensor, torch.Tensor], 
                    mode="updating_model"):
        """
        Calculate the radius from the grid. 
        
        rmaj = |cos(θ)(x - xc)| + |sin(θ)(y - yc)|
        rmin = |- sin(θ)(x - xc)| + |cos(θ)(y - yc)|
        
        r = (rmaj**(B+2) + (rmin/q)**(B+2))**(1/(B+2))

        Parameters
        ---------
        grid
            The grid, transit to galaxy-centered and revise the value close 
            to zero by the decorator.
        mode
            Determine which kind of values will be called in 
            parameter object.  
        """   
              
        y_grid = grid[0]
        x_grid = grid[1]  
        self.x_maj = torch.abs(x_grid * torch.cos(self.pa.value(mode))
                                + y_grid * torch.sin(self.pa.value(mode)))
        self.x_min = torch.abs(- x_grid * torch.sin(self.pa.value(mode)) 
                               + y_grid * torch.cos(self.pa.value(mode)))
        self.radius = torch.pow(torch.pow(self.x_maj, 
                                          self.box.value(mode) + 2) 
                                + torch.pow((self.x_min / self.axis_r.value(mode)), 
                                            (self.box.value(mode) + 2)), 
                                1 / (self.box.value(mode) + 2))

        return self.radius
 
    def image_via_grid_from(
        self, 
        grid: Union[torch.tensor, torch.tensor], 
        mode: str = "updating_model"):
        """
        Returns the light profile's image calculated from a 2D grid. In 
        radial profiles, the profiles are equations with radial, so grid
        need to transit to radial to do the further calculations.

        Parameters
        ----------
        grid
            The galaxy-centered grid.
        mode
            Determine which kind of values will be called in 
            parameter object.

        Returns
        -------
        image
            The model image calculated with the grid.
        """
        return self.image_via_radii_from(self.make_radius(grid, mode), mode)     
     
    def image_via_radii_from(
        self, 
        radius: torch.tensor, 
        mode: str):
        """
        Returns the light profile's image calculated from a radius matrix. 

        Parameters
        ----------
        radius
            The radius of the equivalent circle.
        mode
            Determine which kind of values will be called in 
            parameter object.

        Returns
        -------
        image
            The model image calculated with the radius.
        """
        raise NotImplementedError()

