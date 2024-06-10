from galmoss.Parameter.basic import Parameters
import torch
from typing import Union
import numpy as np
from galmoss.profile.basic import LightProfile, Ellip


class Sersic(Ellip, LightProfile):
    def __init__(
    self, 
    cen_x: Parameters, 
    cen_y: Parameters, 
    pa: Parameters,
    axis_r: Parameters,
    
    eff_r: Parameters,
    mag: Parameters,
    ser_n: Parameters,
    
    box: Parameters = None,
    ):
        """
        The Sersic profile. 
        
        Parameters
        ----------
        cen_x
            The galaxy center in x-xis in arc-second.
        cen_y
            The galaxy center in y-xis in arc-second.
        pa
            The position angle of the ellipse.
        axis_r
            The axis ratio of the ellipse.      
        eff_r
            The effective radius, which encompasses half of the total profile flux.
        mag
            The total magnitude of the galaxy.
        ser_n
            The sersic index, which dictates the profile's curvature
        box
            The boxness of the ellipse.  
        """  
        super().__init__(cen_x=cen_x, cen_y=cen_y, pa=pa, axis_r=axis_r, box=box)
        
        self.eff_r = eff_r
        self.mag = mag
        self.ser_n = ser_n 

    def sersic_constant(self, ser_n: torch.tensor) -> float:
        """
        A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's total integrated light.
        """
        greater_condition = ser_n > 0.36
        greater_values = (
            (2 * ser_n)
            - (1.0 / 3.0)
            + (4.0 / (405.0 * ser_n))
            + (46.0 / (25515.0 * ser_n**2))
            + (131.0 / (1148175.0 * ser_n**3))
            - (2194697.0 / (30690717750.0 * ser_n**4))
        )

        lesser_values = (
            0.01945 
            - 0.8902 * ser_n
            + 10.95 * ser_n**2
            - 19.67 * ser_n**3
            + 13.43 * ser_n**4
        )

        return torch.where(greater_condition, greater_values, lesser_values)

    
    def image_via_radii_from(self, 
                             radius:torch.tensor, 
                             mode: str = "updating_model"):
        """
        Returns the model image of the sersic profile from radius.

        Parameters
        ----------
        radius
            The radius of the equivalent circle.
        mode
            Determine which kind of values will be called in parameter 
            object.
        """              
        I_e = (
            (torch.pow(10, (-0.4 * self.mag.value(mode)))*self.R_box(mode))
             / (torch.pow(self.eff_r.value(mode), 2)*2*torch.pi
             * self.ser_n.value(mode)
             * torch.exp(self.sersic_constant(self.ser_n.value(mode)))
             * self.axis_r.value(mode)
             * (torch.exp(torch.lgamma(2 * self.ser_n.value(mode))))  
             / torch.pow(10, 
                         2 * self.ser_n.value(mode) 
                         *torch.log10(self.sersic_constant(self.ser_n.value(mode)))
                         )
             )
             )
        return (I_e 
                * torch.exp(-self.sersic_constant(self.ser_n.value(mode))
                            * ( torch.pow(radius / self.eff_r.value(mode), 
                                          1.0 / self.ser_n.value(mode))
                               - 1)
                            )
                )

    
class ExpDisk(Ellip, LightProfile):
    def __init__(
    self, 
    cen_x: Parameters, 
    cen_y: Parameters, 
    pa: Parameters,
    axis_r: Parameters,
    
    mag: Parameters,
    scale_l: Parameters,
    
    box: Parameters=None,
    ):
        """
        The Exponential disk profile. 

        Parameters
        ----------
        cen_x
            The galaxy center in x-xis in arc-second.
        cen_y
            The galaxy center in y-xis in arc-second.
        pa
            The position angle of the ellipse.
        axis_r
            The axis ratio of the ellipse.      
        mag
            The total magnitude of the galaxy.
        scale_l
            The scale-length of the disk.        
        box
            The boxness of the ellipse.  
        """  
        super().__init__(cen_x=cen_x, cen_y=cen_y, pa=pa, axis_r=axis_r, box=box)
        self.mag = mag
        self.scale_l = scale_l
        
        
    def image_via_radii_from(self, radius, mode="updating_model"):
        """
        Returns the model image of the Exponential disk profile from radius.

        Parameters
        ----------
        radius
            The radius of the equivalent circle.
        mode
            Determine which kind of values will be called in parameter 
            object.
        """              

        I_0 = (
            (torch.pow(10, -0.4 * self.mag.value(mode))
             * self.R_box(mode)*(self.con.value(mode)-1))
            /(2 * torch.pi * self.axis_r.value(mode) 
              * torch.pow(self.scale_l.value(mode), 2)))
        
        return I_0 * torch.exp(-radius/self.scale_l.value(mode))
                        

class Ferrer(Ellip, LightProfile):
    def __init__(
    self, 
    cen_x: Parameters, 
    cen_y: Parameters, 
    pa: Parameters,
    axis_r: Parameters,
    
    mag: Parameters,
    trunc_r: Parameters,
    trunc_a: Parameters,
    trunc_b: Parameters,

    box: Parameters=None,
    ):
        """
        The Ferrer profile. 

        Parameters
        ----------
        cen_x
            The galaxy center in x-xis in arc-second.
        cen_y
            The galaxy center in y-xis in arc-second.
        pa
            The position angle of the ellipse.
        axis_r
            The axis ratio of the ellipse.      
        mag
            The total magnitude of the galaxy.
        trunc_r
            The the outer truncation radius.     
        trunc_a
            The parameters governing the slopes of the core.
        trunc_b
            The parameters governing the slopes of the truncation.                         
        box
            The boxness of the ellipse.  
        """          
        super().__init__(cen_x=cen_x, cen_y=cen_y, pa=pa, axis_r=axis_r, box=box)
        self.mag = mag
        self.trunc_r = trunc_r
        self.trunc_a = trunc_a
        self.trunc_b = trunc_b

    def image_via_radii_from(self, radius, mode="updating_model"):
        """
        Returns the model image of the Ferrer profile from radius.

        Parameters
        ----------
        radius
            The radius of the equivalent circle.
        mode
            Determine which kind of values will be called in parameter 
            object.
        """              
        radius[radius>self.trunc_r.value(mode)] = self.trunc_r.value(mode)
        
        I_0 = (
            (
                torch.pow(10, (-0.4 * self.mag.value(mode)))
                * self.R_box(mode)
            )
               / (
                   torch.exp(torch.lgamma(self.trunc_a.value(mode))) 
                 * torch.exp(torch.lgamma(1 + 2/(2 - self.trunc_b.value(mode)))) 
                 / torch.exp(torch.lgamma(self.trunc_a.value(mode)
                                          + 1 
                                          + 2 / (2 - self.trunc_b.value(mode))
                                          )
                             ) 
                 * self.trunc_a.value(mode) 
                 * torch.pow(self.trunc_r.value(mode), 2) 
                 * self.axis_r.value(mode)
                 *torch.pi
                 )
               )
        return I_0 * torch.pow(1 - torch.pow(radius/self.trunc_r.value(mode), 
                                             2 - self.trunc_b.value(mode)
                                             ),
                               self.trunc_a.value(mode)
                               )
  
    
class King(Ellip, LightProfile):
    def __init__(
    self, 
    cen_x: Parameters, 
    cen_y: Parameters, 
    pa: Parameters,
    axis_r: Parameters,
    
    inten: Parameters,
    core_r: Parameters,
    trunc_r: Parameters,
    pow_n: Parameters,
    
    box: Parameters=None,
    ):
        """
        The King profile. 

        Parameters
        ----------
        cen_x
            The galaxy center in x-xis in arc-second.
        cen_y
            The galaxy center in y-xis in arc-second.
        pa
            The position angle of the ellipse.
        axis_r
            The axis ratio of the ellipse.      
        inten
            The central surface brightness parameter.
        core_r
            Core radius, signifies the scale at which the density starts 
            to deviate from uniformity.     
        trunc_r
            Truncation radius, marks the boundary of the cluster.
        pow_n
            The global power-law factor, dictates the rate at which the 
            density declines with distance from the center.             
        box
            The boxness of the ellipse.  
        """               
        super().__init__(cen_x=cen_x, cen_y=cen_y, pa=pa, axis_r=axis_r, box=box)
        self.inten = inten
        self.core_r = core_r
        self.trunc_r = trunc_r
        self.pow_n = pow_n


    def image_via_radii_from(self, radius, mode="updating_model"):
        """
        Returns the model image of the King profile from radius.

        Parameters
        ----------
        radius
            The radius of the equivalent circle.
        mode
            Determine which kind of values will be called in parameter 
            object.
        """              
        radius[radius>self.trunc_r.value(mode)] = self.trunc_r.value(mode)
        partA = torch.pow(
            (1 + torch.pow(
                 (self.trunc_r.value(mode)/self.core_r.value(mode)), 
                 2
                 )
             ), 
            1 /self.pow_n.value(mode)
            )
        partB = torch.pow(
            (1 + torch.pow(
                (radius/self.core_r.value(mode)), 
                2
                )
             ), 
            1 / self.pow_n.value(mode)
            )

        return (self.inten.value(mode) 
                * torch.pow(1 - (1/partA), -self.pow_n.value(mode)) 
                * torch.pow(1/partB - 1/partA, self.pow_n.value(mode)))


class Gaussian(Ellip, LightProfile):
    def __init__(
    self, 
    cen_x: Parameters, 
    cen_y: Parameters, 
    pa: Parameters,
    axis_r: Parameters,
    
    inten: Parameters,
    fwhm: Parameters,
    
    box: Parameters=None,
    ):
        """
        The Gaussian profile. 

        Parameters
        ----------
        cen_x
            The galaxy center in x-xis in arc-second.
        cen_y
            The galaxy center in y-xis in arc-second.
        pa
            The position angle of the ellipse.
        axis_r
            The axis ratio of the ellipse.      
        inten
            The central surface brightness parameter.
        fwhm
            The Full Width at Half Maximum.                 
        box
            The boxness of the ellipse.  
        """           
        super().__init__(cen_x=cen_x, cen_y=cen_y, pa=pa, axis_r=axis_r, box=box)
        self.fwhm = fwhm
        self.inten = inten
    

    def image_via_radii_from(self, radius, mode="updating_model"):
        """
        Returns the model image of the Gaussion profile from radius.

        Parameters
        ----------
        radius
            The radius of the equivalent circle.
        mode
            Determine which kind of values will be called in parameter 
            object.
        """                  
        return (self.inten.value(mode) 
                * torch.exp(
                    -torch.pow(radius, 2)
                    / ((2 / 2.354**2) * torch.pow(self.fwhm.value(mode), 2))
                    )
                )
                        

class Moffat(Ellip, LightProfile):
    def __init__(
    self, 
    cen_x: Parameters, 
    cen_y: Parameters, 
    pa: Parameters,
    axis_r: Parameters,
    
    mag: Parameters,
    con: Parameters,
    fwhm: Parameters,
    
    box: Parameters=None,
    ):
        """
        The Moffat profile. 

        Parameters
        ----------
        cen_x
            The galaxy center in x-xis in arc-second.
        cen_y
            The galaxy center in y-xis in arc-second.
        pa
            The position angle of the ellipse.
        axis_r
            The axis ratio of the ellipse.      
        eff_r
            The effective radius, which encompasses half of the total profile flux.
        mag
            The total magnitude of the galaxy.
        con
            The concentration index, dictates whether the distribution is 
            more Lorentzian-like (n =1) or Gaussian-like (n → ∞).
        fwhm
            The sersic index, which dictates the profile's curvature            
        box
            The boxness of the ellipse.  
        """  
        super().__init__(cen_x=cen_x, cen_y=cen_y, pa=pa, axis_r=axis_r, box=box)
        self.mag = mag
        self.con = con
        self.fwhm = fwhm
        
        
    def image_via_radii_from(self, radius, mode="updating_model"):
        """
        Returns the model image of the Moffat profile from radius.

        Parameters
        ----------
        radius
            The radius of the equivalent circle.
        mode
            Determine which kind of values will be called in parameter 
            object.
        """              
        Rd = (self.fwhm.value(mode) 
              / (2 * torch.pow(
                  torch.pow(
                      2, 1 / self.con.value(mode)
                      ) - 1, 0.5
                  )))
        I_e = (torch.pow(10, -0.4 * self.mag.value(mode))
                * self.R_box(mode)*(self.con.value(mode)-1)
               / (torch.pi * self.axis_r.value(mode) * torch.pow(Rd, 2))
               )
        
        return I_e * torch.pow(1 + torch.pow(radius / Rd, 2), 
                               -self.con.value(mode))
                        


class Sky(LightProfile):
    def __init__(
    self, 
    sky_bg: Parameters,
    ):  
        """
        The flat sky light profile. In defalt, this profile will not be
        convolved by PSF image.

        Parameters
        ----------
        sky_bg
            The sky mean value across all pixels without a radial matrix
             
        """     
        super().__init__()
        self.sky_bg = sky_bg
        self.psf = False

    def image_via_grid_from(self, 
                            grid: Union[torch.tensor, torch.tensor], 
                            mode="updating_model"):
        """
        Returns the model image of the flat sky light profile from a grid.

        Parameters
        ----------
        grid
            The coordinate grid (galaxy-centered).
        """        
        return torch.ones_like(grid[0]) * self.sky_bg.value(mode)

   


                        
