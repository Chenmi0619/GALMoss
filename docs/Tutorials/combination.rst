.. _combination_fitting:

Combination profiles
=====================

Here, we demonstrate how to use a combination of two Sérsic profiles to make disk and bulge decomposition on SDSS image data using the **GalMOSS** package.

Load necessary packages
-------------------------

.. code-block:: python
    
    import Galmoss as gm


Define parameter and profile objects 
------------------------------------

Upon importing the package, the subsequent step entails defining parameter objects. To ensure that the center parameter within both profiles remains the same, it suffices to specify the center parameter once and subsequently incorporate it into various profiles.

.. code-block:: python

    xcen = gm.p([65.97, 65.73])
    ycen = gm.p([65.30, 64.81])

For a quick start, we let the disk and bulge profile share the initial value from the SExtractor, with an initial Sérsic index of 1 for the bulge component and 4 for the disk component.

.. code-block:: python

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


Define dataset objects, start fitting 
---------------------------------------

Compared to the single profile case, we only need to change the code of profile definition. We choose to use bootstrap to calculate the uncertainty here.                        

.. code-block:: python

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
