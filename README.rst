
.. image:: repo/logo.jpg
   :alt: 

Why you choose GalMOSS?  
===================================
**8000 galaxies fitting, in only 10 mins!**

**GalMOSS** a Python-based, Torch-powered tool for two-dimensional fitting of galaxy profiles. By seamlessly enabling GPU parallelization, **GalMOSS** meets the high computational demands of large-scale galaxy surveys, placing galaxy profile fitting in the LSST-era. It incorporates widely used profiles such as the Sérsic, Exponential disk, Ferrer, King, Gaussian, and Moffat profiles, and allows for the easy integration of more complex models. 

How to install 
===============
We provide two kinds of methods to download and install **GalMOSS**, pip and git.

Install via pip
---------------

.. code-block:: bash

    pip install autogalaxy


Install via git
---------------

.. code-block:: bash

    git clone https://github.com/Chenmi0619/GALMoss
    cd galmoss
    python setup.py install

You can access it by clicking on   `GitHub-GALMoss <https://github.com/Chenmi0619/GALMoss>`_.


How to use
===========

Fit sersic profile on single galaxy
------------------------------------

Here, we demonstrate how to fit a single Sérsic profile to `SDSS image data <https://github.com/Chenmi0619/GALMoss/tree/main/repo/dataset>`_ using the GALMoss package.

First, we need to load the necessary packages.

.. code-block:: bash

    import Galmoss as gm


Next, we need to define the parameter objects and associate them with profile instances. The initial estimates of the galaxy parameters are provided by \texttt{sextractor}. Notably, we do not include the boxiness parameter in this simple example, despite its availability within the **GalMOSS** framework.

.. code-block:: bash

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




The comprehensive dataset object can be formulated utilising the image sets (galaxy image, mask image, PSF image, sigma image) together with the chosen profiles.

.. code-block:: bash

    dataset = gm.Dataset(
        galaxy_index="J162123.19+322056.4",
        image_path="./J162123.19+322056.4_image.fits",
        sigma_path="./J162123.19+322056.4_sigma.fits",
        psf_path="./J162123.19+322056.4_psf.fits",
        mask_path="./J162123.19+322056.4_mask.fits"
        mask_index=2,
        img_block_path="./test_repo",
        result_path="./test_repo"    
    )

    dataset.define_profiles(sersic=sersic)



After initializing the hyperparameter during the fitting process, training could start. Subsequently, we run the uncertainty estimation process.

.. code-block:: bash

    fitting = gm.Fitting(dataset=dataset, 
                        batch_size=1, 
                        iteration=1000)
    fitting.fit()
    fitting.uncertainty(method="covar_mat")



When the fitting process is completed, the fitted results and the img\_blocks are saved in corresponding path.

Fit bulge\+disk profile on multiple galaxies
---------------------------------------------

Here, we demonstrate how to use a combination of two Sérsic profiles to make disk and bulge decomposition on SDSS image data using the **GalMOSS** package.


.. code-block:: bash
    
    import Galmoss as gm

Upon importing the package, the subsequent step entails defining parameter objects. To ensure that the center parameter within both profiles remains the same, it suffices to specify the center parameter once and subsequently incorporate it into various profiles.

.. code-block:: bash

    xcen = gm.p([65.97, 65.73])
    ycen = gm.p([65.30, 64.81])

For a quick start, we let the disk and bulge profile share the initial value from the SExtractor, with an initial Sérsic index of 1 for the bulge component and 4 for the disk component.

.. code-block:: bash

    bulge = gm.lp.Sersic(cen_x=xcen, 
                        cen_y=ycen, 
                        pa=gm.p([58.7, -8.44], angle=True), 
                        axis_r=gm.p([0.75, 0.61709153]), 
                        eff_r=gm.p([4.09, 18], pix_scale=0.396), 
                        ser_n=gm.p([4, 4], log=True), 
                        mag=gm.p([17.97, 15.6911], M0=22.5))

    disk = gm.lp.Sersic(cen_x=xcen, 
                        cen_y=ycen, 
                        pa=gm.p([58.7, -8.44], angle=True), 
                        axis_r=gm.p([0.75, 0.61709153]),  
                        eff_r=gm.p([4.09, 18], pix_scale=0.396), 
                        ser_n=gm.p([1, 1], log=True), 
                        mag=gm.p([17.97, 15.6911], M0=22.5))

Compared to the single profile case, we only need to change the code of profile definition. We choose to use bootstrap to calculate the uncertainty here.                        

.. code-block:: bash

    dataset = gm.DataSet(["J100247.00+042559.8", "J092800.99+014011.9"],
                    image_path=["./J100247.00+042559.8_image.fits",
                                "./J092800.99+014011.9_image.fits"],
                    sigma_path=["./J100247.00+042559.8_sigma.fits",
                                "./J092800.99+014011.9_sigma.fits"],
                    psf_path=["./J100247.00+042559.8_psf.fits",
                            "./J092800.99+014011.9_psf.fits"],
                    mask_path=["./J100247.00+042559.8_mask.fits", 
                            "./J092800.99+014011.9_mask.fits"],
                    img_block_path="./test_repo/",
                    result_path="./test_repo/"
    )
    dataset.define_profiles(bulge=bulge, disk=disk)
    fitting = gm.Fitting(dataset=dataset, 
                        batch_size=1, 
                        iteration=1000)
    fitting.fit()
    fitting.uncertainty(method="bstrap")


Requirements
=============
numpy>=1.21.0

pandas>=1.4.4

torch>=2.0.1

astropy>=5.1

h5py>=3.7.0

torch-optimizer>=0.3.0

tqdm>=4.64.1

   