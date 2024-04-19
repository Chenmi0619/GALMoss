import numpy as np
import torch
from functools import wraps

def inv_from_normalization(func):
    """
    Workflow of parameters in galmoss:
        Original domain 
            --> to_mapped --> 
        mapped domain 
            --> to_logged --> 
        logged domain
            --> to_normalized -->
        normalized domain and be load into optimizer
    The inv-workflow follows the same logic.

    Checks whether need to transform input values from normlization to 
    logged domain follows the attribute `value_range`. If value_range 
    is not None, which means the value in the func has already transformed 
    into (-1, 1) domain before, so the transition will be effective. 
    Otherwise, return input values. 

    Parameters
    ----------
    func: (parameter, *args, **kwargs) -> torch.tensor
        A function that receives values in (-1, 1)/non-normalized domain.

    Returns
    -------
        A function that can accept normalized or non-normalized values.
    """
    @wraps(func)
    def wrapper(
        cls,
        normalized_value: torch.tensor,
        *args,
        **kwargs
    ) -> torch.tensor:
        """
        Parameters
        ----------
        cls
            The class that owns the function.
            
        normalized_value
            The parameter value (maybe in (-1, 1) domain, depends on the
             `log` attribution).

        Returns
        -------
            The parameter value in mapped domain after conversion.
        """    
        value_range = cls.value_range #param_value_range
        
        if value_range != (None, None):
            return func(
                cls, 
                ((
                    torch.tanh(normalized_value) 
                        - cls.normalization_relationship[1]
                 ) 
                    * cls.normalization_relationship[0]
                ), 
                *args, 
                **kwargs
            ) 
        else:
          return func(cls, normalized_value, *args, **kwargs)
      
    return wrapper


def inv_from_log(func):
    """
    Workflow of parameters in galmoss:
        Original domain 
            --> to_mapped --> 
        mapped domain 
            --> to_logged --> 
        logged domain
            --> to_normalized -->
        normalized domain and be load into optimizer
    The inv-workflow follows the same logic.

    Checks whether need to transform input values from logged to 
    mapped domain follows the attribute `log`. If log is True, which 
    means the value in the func has already transformed into logarithmic 
    domain before, so the transition will be effective. Otherwise, return 
    input values.

    Parameters
    ----------
    func: (parameter, *args, **kwargs) -> torch.tensor
        A function that receives values in logarithmic/mapped domain.

    Returns
    -------
        A function that can accept logged or mapped values.
    """  
    @wraps(func)
    def wrapper(
        cls, 
        logged_value: torch.tensor, 
        *args,
        **kwargs
    ) -> torch.tensor:  
        """
        Parameters
        ----------
        cls
            The class that owns the function.
            
        logged_value
            The parameter value (maybe in logarithmic domain, depends on the
             `log` attribution).

        Returns
        -------
            The parameter value in normal domain after conversion.
        """             
        if cls.log:  
            return func(cls, torch.pow(10, logged_value), *args, **kwargs)  
        else:
            return func(cls, logged_value, *args, **kwargs)
           
    return wrapper


   
def inv_from_mapping_uncertainty(func):
    """
    Workflow of parameters in galmoss:
        Original domain 
            --> to_mapped --> 
        mapped domain 
            --> to_logged --> 
        logged domain
            --> to_normalized -->
        normalized domain and be load into optimizer
    The inv-workflow follows the same logic.

    Checks whether need to transform input values from mapped to 
    original domain follows the attribute `M0`, `pix_scale` and `angle`. 
    If M0 and pix_scale is not None or angle is True, which means the 
    value in the func has already transformed into mapped domain before, 
    so the transition will be effective. Otherwise, return input values. 

    Parameters
    ----------
    func: (parameter, *args, **kwargs) -> torch.tensor
        A function that receives uncertainty in mapped/non-mapped domain.

    Returns
    -------
        A function that can accept mapped or non-mapped uncertainty.
    """
    @wraps(func)
    def wrapper(cls, *args, **kwargs) -> torch.tensor:
        """
        Parameters
        ----------
        cls
            The class that owns the function.

        Returns
        -------
            The value attribute in original domain.
        """            
        mapped_uc = func(cls, *args, **kwargs)

        if cls.angle:
            inv_mapped_uc = mapped_uc * 180 / torch.pi
        elif cls.pix_scale is not None:
            inv_mapped_uc = mapped_uc * cls.pix_scale_device.squeeze()
        else:
            inv_mapped_uc = mapped_uc
            
        return inv_mapped_uc

    return wrapper
    
    
def inv_from_mapping(func):
    """
    Workflow of parameters in galmoss:
        Original domain 
            --> to_mapped --> 
        mapped domain 
            --> to_logged --> 
        logged domain
            --> to_normalized -->
        normalized domain and be load into optimizer
    The inv-workflow follows the same logic.

    Checks whether need to transform input values from mapped to 
    original domain follows the attribute `M0`, `pix_scale` and `angle`. 
    If M0 and pix_scale is not None or angle is True, which means the 
    value in the func has already transformed into mapped domain before, 
    so the transition will be effective. Otherwise, return input values. 

    Parameters
    ----------
    func: (parameter, *args, **kwargs) -> torch.tensor
        A function that receives values in mapped/non-mapped domain.

    Returns
    -------
        A function that can accept mapped or non-mapped values.
    """
    @wraps(func)
    def wrapper(cls, *args, **kwargs) -> torch.tensor:
        """
        Parameters
        ----------
        cls
            The class that owns the function.

        Returns
        -------
            The value attribute in original domain.
        """            
        mapped_value = func(cls, *args, **kwargs)
        if cls.M0 is not None:
            inv_mapped_value = mapped_value + cls.M0
        elif cls.angle:
            inv_mapped_value = mapped_value * 180 / torch.pi
            inv_mapped_value = transiform_eff_degrees(inv_mapped_value)
        elif cls.pix_scale is not None:
            inv_mapped_value = mapped_value * cls.pix_scale_device
        else:
            inv_mapped_value = mapped_value
        return inv_mapped_value

    return wrapper


def to_log(func): 
    """
    Workflow of parameters in galmoss:
        Original domain 
            --> to_mapped --> 
        mapped domain 
            --> to_logged --> 
        logged domain
            --> to_normalized -->
        normalized domain and be load into optimizer
    The inv-workflow follows the same logic.

    Checks whether need to transform input values to logged domain 
    follows the attribute `log`. If log is True, the transition will be 
    effective. Otherwise, return input values.

    Parameters
    ----------
    func: (parameter, *args, **kwargs) -> torch.tensor
        A function that returns value attributes, which possibly needs log
        transformation.

    Returns
    -------
        A function that returns a value attribute.
    """     
    @wraps(func)
    def wrapper(cls, *args, **kwargs) -> torch.tensor:
        """
        Parameters
        ----------
        cls
            The class that owns the function.

        Returns
        -------
            The value attribute in logarithmic domain (cls.log=True), 
            normal domain (cls.log=False), or None if value attributes 
            is None.
        """    
        Params = func(cls, *args, **kwargs)
        if Params is not None:
            if cls.log:  
                return torch.log10(Params)   
            else:
                return Params
        else:
            return None
        
    return wrapper
        
        
def to_normalization(func):
    """
    Workflow of parameters in galmoss:
        Original domain 
            --> to_mapped --> 
        mapped domain 
            --> to_logged --> 
        logged domain
            --> to_normalized -->
        normalized domain and be load into optimizer
    The inv-workflow follows the same logic.

    Checks whether need to normalize input values into (-1, 1) noramlized 
    domain follows the attribute `value_range`. If value_range is not None, 
    which means the value has constrains during the fitting pocess, so the 
    transition will be effective. Otherwise, return input values.    

    Parameters
    ----------
    func: (parameter, *args, **kwargs) -> torch.tensor
        A function that returns value attributes, which possibly needs 
        to be normalizationed.

    Returns
    -------
        A function that returns a value attribute.
    """
    @wraps(func)
    def wrapper(cls, *args, **kwargs) -> torch.tensor:
        """
        Parameters
        ----------
        cls: 
            The class that owns the function.

        Returns
        -------
            The value attribute in (-1, 1) domain (cls.value_range is 
            not None), normal domain (cls.value_range is None), or None 
            if value attributes is None.
        """
        
        Params = func(cls, *args, **kwargs)
        value_range = cls.value_range
        if Params is not None:
            if value_range != (None, None):  
                return (
                    torch.atanh(
                        (Params - value_range[1]) 
                        / (0.5 * (value_range[1] - value_range[0]))
                        + 1                       
                    )
                )
            else:
                return Params
        else:
            return None

    return wrapper


def to_mapping_log_with_value_range(func):
    """
    Workflow of parameters in galmoss:
        Original domain 
            --> to_mapped --> 
        mapped domain 
            --> to_logged --> 
        logged domain
            --> to_normalized -->
        normalized domain and be load into optimizer
    The inv-workflow follows the same logic.

    Checks whether need to transform input values from original to 
    mapped and then to logged domain. 

    Parameters
    ----------
    func: (parameter, *args, **kwargs) -> torch.tensor
        A function that receives values and value ranges in original 
        domain.

    Returns
    -------
        A function that can accept original values and value ranges.
    """
    @wraps(func)
    def wrapper(
        cls, 
        parameters,
        value_range,
        M0,
        pix_scale,
        angle,
        *args, 
        **kwargs
    ) -> torch.tensor:
        """

        Parameters
        ----------
        cls
            The class that owns the function.
            
        parameters
            The parameter value in original domain.

        value_range
            The parameter value in original domain.

        M0
            The magnitude zero point.

        pix_scale
            The arcsec to pixel ratio.

        angle
            To show whether the parameter is an angle or not.

        Returns
        -------
            The parameter value in logged domain after conversion.
        """   
        if value_range != (None, None):
            if M0 is not None:
                mapped_value_range = [value_range[0] - M0, 
                                      value_range[1] - M0]
            elif angle:
                mapped_value_range = [value_range[0] * torch.pi / 180, 
                                      value_range[1] * torch.pi / 180]
            elif pix_scale is not None:
                mapped_value_range = [value_range[0] / pix_scale, 
                                      value_range[1] / pix_scale]
            else:
                mapped_value_range = value_range
        else:
            mapped_value_range = value_range

        if M0 is not None:
            mapped_parameters = parameters - M0
        elif angle:
            mapped_parameters = (transiform_eff_degrees(parameters) 
                                 * torch.pi / 180)
        elif pix_scale is not None:
            if isinstance(pix_scale, (int, float)):
                mapped_parameters = parameters / pix_scale
            else:
                mapped_parameters = parameters / pix_scale.reshape(-1, 1)
        else:
            mapped_parameters = parameters                      

        if cls.log:
            log_parameters = np.log10(mapped_parameters)
            if mapped_value_range == (None, None):
                log_value_range = mapped_value_range
            else:
                log_value_range = (np.log10(mapped_value_range[0]), 
                                   np.log10(mapped_value_range[1]))
        else:
            log_parameters = mapped_parameters
            log_value_range = mapped_value_range

        return func(cls, log_parameters, log_value_range, *args, **kwargs)          

    return wrapper


def redim(func):
    """
    To transit input values into torch.tensor, and resize to (param_length, 
    param_dim). param_length equals the number of total galaxies. 

    Parameters
    ----------
    func: (parameter, *args, **kwargs) -> torch.tensor
        A function that receives values in whatever data types.

    Returns
    -------
        A function that can accept whatever data types.
    """
    @wraps(func)
    def wrapper(
        cls, 
        parameters,
        *args, 
        **kwargs
    ) -> torch.tensor:
        """
        Parameters
        ----------
        cls
            The class that owns the function.
            
        parameters
            The parameter value that being input in the object by users.

        Returns
        -------
            The parameter value in torch.tensor, and being resize into 
            (param_length, param_dim).
        """        
        if not isinstance(parameters, torch.Tensor):
            parameters = torch.from_numpy(np.asarray(parameters)).squeeze() 
        else:
            parameters = parameters.squeeze()
        
        if cls.dim == "1d":
            if parameters.ndim not in [0, 1]:
                raise ValueError("Invalid input dimensions for SingleDim mode.")
            parameters = parameters.view(-1, 1) 
        elif cls.dim == "2d":
            if parameters.ndim != 2:
                raise ValueError("Invalid input dimensions for MultiDim mode,"
                                 "expected 2D tensor.")
        else:
            raise ValueError("Invalid mode specified.")

        cls.total_initial_value_cpu_tensor = parameters
        return func(cls, parameters, *args, **kwargs) 
    return wrapper


def transiform_eff_degrees(degrees):
    """
    To map the degrees out of the value range into the effective value 
    inside the value range (-180~180).

    Parameters
    ----------
    degrees
        The degrees, that may have values out of the value range.

    Returns
    -------
        The degrees all inside the value range.    
    """
    degrees = degrees % 360
    degrees = torch.where(degrees > 180, degrees - 360, degrees)
    degrees = torch.where(degrees > 90, degrees - 180, degrees)
    degrees = torch.where(degrees < -90, degrees + 180, degrees)
    return degrees

