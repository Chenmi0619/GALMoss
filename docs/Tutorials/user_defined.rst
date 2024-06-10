.. _user_defined:

User defined profile
=======================

Here, we show how to integrate a more realistic sky profile as an example of a user-defined profile. In this new sky profile, the sky intensity is defined as follows:

.. math::
    I = I_0 + k_{\mathrm{x}}(x - x_0) + k_{\mathrm{y}}(y - y_0).

To integrate this new profile, we need to define a new class, which we have named **NewSky**. This class should include a function that guides the generation of the image model from the given parameters. 

Here are some caveats.  

- every profile class should inherit from the basic class `LightProfile` to access general light profile functions. 
- the parameters should be loaded into the profile class in the `__init__` function, and set as attributions (e.g., `self.sky_0` = `sky_0`).

Define new profile
--------------------

.. code-block:: python

    import galmoss as gm

    class NewSky(gm.LightProfile):
        def __init__(self, sky_0, grad_x, grad_y):
            super().__init__()
            self.psf = False
            self.sky_0 = sky_0
            self.grad_x = grad_x
            self.grad_y = grad_y
        
        def image_via_grid_from(self,
                                grid, 
                                mode="updating_model"):
            return (self.sky_0.value(mode)
                    + self.grad_x.value(mode)
                    * (grid[1]-(grid[0].shape[1] + 1)/2) 
                    + self.grad_y.value(mode)
                    * (grid[0]-(grid[0].shape[0] + 1)/2))

The equation for the profile is defined within the `image_via_grid_from` function. Parameter values are extracted after the mode value, which has a default value of **updating_model**. This mode calls for values that are continuously updated throughout the fitting process and have already been broadcast to a suitable shape for multi-dimensional matrix calculations.


Use new profile
--------------------

A newly defined profile can be used as follows:

.. code-block:: python

    sky = NewSky(sky_0=gm.p(0.3),
                grad_x=gm.p(2),
                grad_y=gm.p(3))