import numpy as np
from typing import Union

from lifesim.core.modules import ElectronNoiseDetectorModule


class ElectronNoiseDarkCurrent(ElectronNoiseDetectorModule):
    """
    This class simulates the dark current noise contribution of the detector to the interferometric
    measurement of LIFE.
    """

    def __init__(self,
                 name: str):
        super().__init__(name=name)
        """
        Parameters
        ----------
        name : str
            Name of the module.
        """

    def noise(self,
              index: Union[int, type(None)]):
        """
        Simulates the amount of electron noise originating from the dark current of the detector
        leaking into the LIFE array measurement.

        Parameters
        ----------
        index: Union[int, type(None)]
            Specifies the planet for which to calculate the noise contribution. If an integer n is
            given, the noise will be calculated for the n-th row in the `data.catalog`. If `None`
            is given, the noise is caluculated for the parameters located in `data.single`.

        Returns
        -------
        dc_leak
            Dark current leakage of the detector in [electron s-1] per wavelength bin.

        Notes
        -----
        All of the following parameters are needed for the calculation of the dark current noise
        contribution and should be specified either in `data.catalog` or `data.single` or 'data.inst' or 'data.options'.

        data.options.array['wl_min'] : float
            Minimum wavelength of the spectrometer in [microns].
        data.options.array['wl_max'] : float
            Maximum wavelength of the spectrometer in [microns].
        data.options.array['spec_res_inst'] : float
            Spectral resolution used for the detector, for simulations with lower spec_res (dimensionless).
        data.inst['wl_bins'] : np.ndarray
            Central values of the spectral bins in the wavelength regime in [m].
        data.options.array['dc_per_pix'] : float
            Dark current per pixel in [electron s-1 px-1].
        """

        # get the wavelength bins for the instrument spectral resolution
        wl_edge = self.data.options.array['wl_min']
        wl_bins = []

        while wl_edge < self.data.options.array['wl_max']:

            # set the wavelength bin width according to the spectral resolution
            wl_bin_width = wl_edge / self.data.options.array['spec_res_inst'] / \
                           (1 - 1 / self.data.options.array['spec_res_inst'] / 2)

            # make the last bin shorter when it hits the wavelength limit
            if wl_edge + wl_bin_width > self.data.options.array['wl_max']:
                wl_bin_width = self.data.options.array['wl_max'] - wl_edge

            # calculate the center and edges of the bins
            wl_center = wl_edge + wl_bin_width / 2
            wl_edge += wl_bin_width

            wl_bins.append(wl_center)
        
        wl_bins = np.array(wl_bins) * 1e-6  # in m

        # read data on detector
        dc_per_pix = self.data.options.array['dc_per_pix']
        res_ratio = len(wl_bins) / len(self.data.inst['wl_bins'])  # ratio of number of bins at spec_res and spec_res_inst
        # total_pixels = self.data.options.array['pix_per_wl'] * self.data.options.array['spec_res_inst'] # minimum number of detector pixels (nyquist rate)

        # calculate total dark current noise
        dc_leak = np.full(self.data.inst['wl_bins'].shape, dc_per_pix * self.data.options.array['pix_per_wl'] * res_ratio)

        return dc_leak