from galmoss.profile.ellip_profile import Ellip
from galmoss.Parameter.parameters import *
from galmoss.Parameter.basic import AbstractParameters
from typing import Union, Tuple
# from fitting import Fitting
import pynvml

from functools import wraps
import numpy as np
import torch
import inspect
device = 'cuda' if torch.cuda.is_available() else 'cpu'

  
class Light_Profile:
    def image_2d_grid_from_(self, grid, mode="updating_value"):
        return self.image_2d_from_(self.make_radius(grid, mode))  
  
    def param(self, name: str) -> dict:
        flex_dic = {}
        fix_dic = {}
        length = []
        for attr_name in dir(self):

            attr = getattr(self, attr_name)

            if isinstance(attr, AbstractParameters):
                length.append(attr.param_length)
                idx = name + "_" + self.__class__.__name__ + "_" + attr.__class__.__name__
                if attr.if_fit:
                    flex_dic[idx] = attr 
                else:
                    fix_dic[idx] = attr
        assert all(x == length[0] for x in length), "the length of parameters are not the same!" 
        return flex_dic, fix_dic
  

        

class Sky(Light_Profile):
    def __init__(
    self, 
    skyBg:SkyBg
    ):
        self.skyBg = skyBg
    
    def make_radius(self, grid, mode):
        return grid[0]
 
    def image_2d_from_(self, radius, mode="updating_value"):
        return torch.ones_like(radius) * self.skyBg.value(mode)
              

      
class Sersic(Ellip, Light_Profile):
    def __init__(
    self, 
    cenX: CenX, 
    cenY: CenY, 
    pa: PA,
    axisR: AxisR,
    
    effR: EffR,
    mag: Mag,
    serN: SerN,
    
    box: Box = None,

    ):

      
        super().__init__(cenX=cenX, cenY=cenY, pa=pa, axisR=axisR, box=box)
        self.effR = effR
        self.mag = mag
        self.serN = serN 

    def sersic_constant(self, serN) -> float:
        """
        A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
        total integrated light.
        """

        return (
            (2*serN)
            - (1.0/3.0)
            + (4.0/(405.0 * serN))
            + (46.0/(25515.0 * serN**2))
            + (131.0/(1148175.0 * serN**3))
            - (2194697.0/(30690717750.0 * serN**4))
        )
    
    def image_2d_from_(self, radius, mode="updating_value"):
        
        I_e = (
            (torch.pow(10, (-0.4 * self.mag.value(mode)))*self.R_box(mode))
             / (torch.pow(self.effR.value(mode), 2)*2*torch.pi
             * self.serN.value(mode)
             * torch.exp(self.sersic_constant(self.serN.value(mode)))
             * self.axisR.value(mode)
             * (torch.exp(torch.lgamma(2 * self.serN.value(mode))))  
             / torch.pow(10, 
                         2 * self.serN.value(mode) 
                         *torch.log10(self.sersic_constant(self.serN.value(mode)))
                         )
             )
             )

    
        return (I_e 
                * torch.exp(-self.sersic_constant(self.serN.value(mode))
                            * ( torch.pow(radius / self.effR.value(mode), 
                                          1.0 / self.serN.value(mode))
                               - 1)
                            )
                )


      
class Moffat(Ellip, Light_Profile):
    def __init__(
    self, 
    cenX: CenX, 
    cenY: CenY, 
    pa: PA,
    axisR: AxisR,
    
    mag: Mag,
    con: Con,
    fwhm: Fwhm,
    
    box: Box,
    ):
        super().__init__(cenX=cenX, cenY=cenY, pa=pa, axisR=axisR, box=box)
        self.mag = mag
        self.con = con
        self.fwhm = fwhm
        
        
    def image_2d_from_(self, radius, mode="updating_value"):
        
        Rd = (self.fwhm.value(mode) 
              / (2*torch.pow(torch.pow(2, 1/self.con.value(mode)) - 1, 0.5)))

        I_e = ((torch.pow(10, -0.4 * self.mag.value(mode))*self.R_box(mode)*(self.con.value(mode)-1))
               /(torch.pi * self.axisR.value(mode) * torch.pow(Rd, 2)))
        
        return I_e * torch.pow(1 + torch.pow(radius/Rd, 2), -self.con.value(mode))
                        
    def image_2d_grid_from_(self, grid, mode="updating_value"):
        
        return self.image_2d_from_(self.make_radius(grid, mode), mode)  

          
      

class Gaussian(Ellip, Light_Profile):
    def __init__(
    self, 
    cenX: CenX, 
    cenY: CenY, 
    pa: PA,
    axisR: AxisR,
    
    inten: Inten,
    disp: Disp,
    box: Box


    ):
        super().__init__(cenX=cenX, cenY=cenY, pa=pa, axisR=axisR, box=box)
        self.disp = disp
        self.inten = inten
    

    def image_2d_from_(self, radius, mode="updating_value"):
            
        return self.inten.value(mode) * torch.exp(-torch.pow(radius, 2)/(2 * torch.pow(self.disp.value(mode), 2)))
                        
    
    def image_2d_grid_from_(self, grid, mode="updating_value"):
        return self.image_2d_from_(self.make_radius(grid, mode), mode)
    

      
      
    
       
    
              
class ModifiedFerrer(Ellip, Light_Profile):
    def __init__(
    self, 
    cenX: CenX, 
    cenY: CenY, 
    pa: PA,
    axisR: AxisR,
    
    
    mag: Mag,
    truncR: TruncR,
    truncA: TruncA,
    truncB: TruncB,

    box: Box

    ):
        super().__init__(cenX=cenX, cenY=cenY, pa=pa, axisR=axisR, box=box)
        self.mag = mag
        self.truncR = truncR
        self.truncA = truncA
        self.truncB = truncB

    def image_2d_from_(self, radius, mode="updating_value"):

        I_0 = ((torch.pow(10, (-0.4 * self.mag.value(mode)))*self.R_box(mode))
               /(torch.exp(torch.lgamma(self.truncB.value(mode))) 
                 * torch.exp(torch.lgamma(1 + 2/(2 - self.truncA.value(mode)))) 
                 / torch.exp(torch.lgamma(self.truncB.value(mode)+ 1 + 2/(2 - self.truncA.value(mode)))) 
                 * self.truncB.value(mode) 
                 * torch.pow(self.truncR.value(mode), 2) 
                 * self.axisR.value(mode)
                 *torch.pi)
               )
        return I_0 * torch.pow(1 - torch.pow(radius/self.truncR.value(mode), 
                                             2 - self.truncA.value(mode)
                                             ),
                               self.truncB.value(mode)
                               )
                        
    
    def image_2d_grid_from_(self, grid, mode="updating_value"):
        return self.image_2d_from_(self.make_radius(grid, mode), mode)
  

      
class King(Ellip, Light_Profile):
    def __init__(
    self, 
    cenX: CenX, 
    cenY: CenY, 
    pa: PA,
    axisR: AxisR,
    
    
    inten: Inten,
    coreR: CoreR,
    truncR: TruncR,
    powN: PowN,
    
    box: Box

    ):
        super().__init__(cenX=cenX, cenY=cenY, pa=pa, axisR=axisR, box=box)
        self.inten = inten
        self.coreR = coreR
        self.truncR = truncR
        self.powN = powN


    def image_2d_from_(self, radius, mode="updating_value"):
        
        radius[radius>self.truncR.value(mode)] = self.truncR.value(mode)
        partA = torch.pow((1+torch.pow((self.truncR.value(mode)/self.coreR.value(mode)), 2)), 1/self.powN.value(mode))
        partB = torch.pow((1+torch.pow((radius/self.coreR.value(mode)), 2)), 1/self.powN.value(mode))

        return (self.inten.value(mode) 
                * torch.pow(1 - (1/partA), -self.powN.value(mode)) 
                * torch.pow(1/partB - 1/partA, self.powN.value(mode)))
                        
