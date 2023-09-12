from galmoss import Fitting
import numpy as np

class Plot:
  def __init__(
  self, 
  image_data,
  model_data,
  mask_data = None,
  scale_data = None,
  psf_data = None,
  sigma_data = None

):
    
    
class Basic_Plot(Plot):
  def __init__(
  self, 
  image_data,
  model_data,
  mask_data = None,
  scale_data = None,
  psf_data = None,
  sigma_data = None

):
    super().__init__(
      image_data=image_data, 
      model_data=model_data, 
      mask_data=mask_data, 
      scale_data=scale_data, 
      psf_data=psf_data,
      sigma_data=sigma_data
      )
  
  
  @classmethod
  def plot_from_fitting(self, fitting: Fitting, galaxy_id: str):
    name = fitting.name
    index = np.where(name == galaxy_id)[0]
    model_data = fitting.model[index]
    image_data = fitting.dataset.read_data(fitting.dataset.image_path, 
                                           galaxy_id,
                                           fitting.dataset.image_hdu)
    mask_data = fitting.dataset.read_data(fitting.dataset.mask_path, 
                                           galaxy_id,
                                           fitting.dataset.mask_hdu)    
    scale_data = fitting.dataset.read_data(fitting.dataset.seg_path, 
                                           galaxy_id,
                                           fitting.dataset.seg_hdu)       
    psf_data = fitting.dataset.read_data(fitting.dataset.psf_path, 
                                           galaxy_id,
                                           fitting.dataset.psf_hdu)    
    sigma_data = fitting.dataset.read_data(fitting.dataset.sigma_path, 
                                           galaxy_id,
                                           fitting.dataset.sigma_hdu)        
    return  Basic_Plot(
          image_data = image_data,
          model_data = model_data,
          mask_data = mask_data,
          scale_data = scale_data,
          psf_data = psf_data,
          sigma_data = sigma_data
        )
    
