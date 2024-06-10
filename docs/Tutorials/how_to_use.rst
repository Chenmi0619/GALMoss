.. _easy_start:

An easiest start
==================

Here, we demonstrate how to fit a single Sérsic profile to `SDSS image data <https://github.com/Chenmi0619/GALMoss/tree/main/repo/dataset>`_ using the GALMoss package.



Load necessary packages
-------------------------

First, we need to load the necessary packages.

.. code-block:: python

    import Galmoss as gm


Define parameter and profile objects 
------------------------------------

Next, we need to define the parameter objects and associate them with profile instances. The initial estimates of the galaxy parameters are provided by \texttt{sextractor}. Notably, we do not include the boxiness parameter in this simple example, despite its availability within the **GalMOSS** framework.

.. code-block:: python

    # define parameter objects and profile

    sersic = gm.lp.Sersic(
        cen_x=gm.p(65.43),
        cen_y=gm.p(64.95),
        pa=gm.p(-81.06, angle=True), 
        axis_r=gm.p(0.64),
        eff_r=gm.p(7.58, pix_scale=0.396),
        ser_n=gm.p(1.53, log=True),
        mag=gm.p(17.68, M0=22.5)
    )



Define dataset objects 
-----------------------

The comprehensive dataset object can be formulated utilising the image sets (galaxy image, mask image, PSF image, sigma image) together with the chosen profiles.

.. code-block:: python

    dataset = gm.DataSet(
        galaxy_index="J162123.19+322056.4",
        image_path="./J162123.19+322056.4_image.fits",
        sigma_path="./J162123.19+322056.4_sigma.fits",
        psf_path="./J162123.19+322056.4_psf.fits",
        mask_path="./J162123.19+322056.4_mask.fits",
        mask_index=2,
        img_block_path="./test_repo",
        result_path="./test_repo"    
    )

    dataset.define_profiles(sersic=sersic)


Start training 
---------------

After initializing the hyperparameter during the fitting process, training could start. Subsequently, we run the uncertainty estimation process.

.. code-block:: python

    fitting = gm.Fitting(dataset=dataset, 
                        batch_size=1, 
                        iteration=1000)
    fitting.fit()
    fitting.uncertainty(method="covar_mat")



When the fitting process is completed, the fitted results and the img\_blocks are saved in corresponding path.