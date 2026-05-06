import os
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
import seaborn as sns
from pathlib import Path

import lifesim
from lifesim.util.habitable import single_habitable_zone

class YieldAnalysis:
    
    def interpolate_diameter(self, 
                             source_path):
        subdirs = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
        diams_float = [float('.'.join(d.split('_')[1:])) for d in subdirs]
        diams = ['_'.join(d.split('_')[1:]) for d in subdirs]

        # uu = upper uncertainty, lu = lower uncertainty
        mission_time = pd.DataFrame(index=np.sort(diams_float), columns=['mission_time_mean', 'lu_mission_time', 'uu_mission_time', 'mission_time_opt_factor'])

        for subdir, d in zip(subdirs, diams_float):
            data_path = os.path.join(source_path, subdir)
            # read the csv file in data path that ends with _mission_time.csv, return error if multiple or none found
            csv_files = [f for f in os.listdir(data_path) if f.endswith('_mission_time.csv')]
            if len(csv_files) != 1:
                raise ValueError(f'Expected one _mission_time.csv file in {data_path}, found {len(csv_files)}')
            data = pd.read_csv(os.path.join(data_path, csv_files[0]))
            mission_time.loc[d, 'mission_time_mean'] = data['total'].mean()
            mission_time.loc[d, 'lu_mission_time'] = data['total'].mean() - data['total'].quantile(0.16)
            mission_time.loc[d, 'uu_mission_time'] = data['total'].quantile(0.84) - data['total'].mean()
            det_ratio = data['detection'] / (data['total'])
            mission_time.loc[d, 'detection_ratio_mean'] = det_ratio.mean()
            mission_time.loc[d, 'lu_detection_ratio'] = det_ratio.mean() - det_ratio.quantile(0.16)
            mission_time.loc[d, 'uu_detection_ratio'] = det_ratio.quantile(0.84) - det_ratio.mean()

            if 'n_Experiment_1' in data.columns:
                mission_time.loc[d, 'mission_time_opt_factor'] = data.loc[data['n_Experiment_1'] == data['n_Experiment_1'].max(), 'total'].mean()
            else:
                mission_time.loc[d, 'mission_time_opt_factor'] = data.loc[data['n_Experiment_2'] == data['n_Experiment_2'].max(), 'total'].mean()

        max_times = np.arange(0.5, 20.1, 0.5)

        max_time_table = pd.DataFrame(index=np.sort(max_times), columns=['diameter_mean', 'lu_diameter', 'uu_diameter', 'diameter_opt_factor'])

        n = len(mission_time)
        kind = 'cubic' if n >= 4 else 'quadratic' if n >= 3 else 'linear'
        # make a spline interpolation of all data points in mission time data frame and evaluate at max_times
        spline = interp1d(np.array(mission_time['mission_time_mean'] / 365.25 / 24 / 60 / 60, dtype=float), np.array(mission_time.index, dtype=float), kind=kind, fill_value='extrapolate')
        max_time_table['diameter_mean'] = spline(max_times)
        # for uncertainties, do linear interpolation of upper and lower bounds
        spline_lu = interp1d(np.array((mission_time['mission_time_mean'] - mission_time['lu_mission_time']) / 365.25 / 24 / 60 / 60, dtype=float), np.array(mission_time.index, dtype=float), kind='linear', fill_value='extrapolate')
        spline_uu = interp1d(np.array((mission_time['mission_time_mean'] + mission_time['uu_mission_time']) / 365.25 / 24 / 60 / 60, dtype=float), np.array(mission_time.index, dtype=float), kind='linear', fill_value='extrapolate')
        max_time_table['lu_diameter'] = max_time_table['diameter_mean'] - spline_lu(max_times)
        max_time_table['uu_diameter'] = spline_uu(max_times) - max_time_table['diameter_mean']
        # for optimal factor diameters
        spline_opt = interp1d(np.array(mission_time['mission_time_opt_factor'] / 365.25 / 24 / 60 / 60, dtype=float), np.array(mission_time.index, dtype=float), kind=kind, fill_value='extrapolate')
        max_time_table['diameter_opt_factor'] = spline_opt(max_times)

        # save max time table to csv
        max_time_table.to_csv(os.path.join(source_path, source_path.split('/')[-1] + '_diameter_mission_time.csv'))
        mission_time.to_csv(os.path.join(source_path, source_path.split('/')[-1] + '_diameter_mission_time_RAW.csv'))

    def run_interpolation(self, 
                          path):
        # run the interpolation on all opt_ directories in the given path
        # path = "/home/kirschkobold/documents/life_internship/LIFEsim_yields/euler_data"

        # find all subdirectories (and subsub, and so on) in path that start with 'opt_'
        opt_dirs = []
        for root, dirs, files in os.walk(path):
            for dirname in dirs:
                if dirname.startswith('opt_'):
                    opt_dirs.append(os.path.join(root, dirname))

        for opt_dir in opt_dirs:
            self.interpolate_diameter(opt_dir)

    def get_eff_eta(self, 
                    catalog_path,
                    config_path="/home/kirschkobold/documents/life_internship/LIFEsim_yields/optimizer_config.yaml"):
        bus = lifesim.Bus()

        # setting the options
        bus.build_from_config(filename=config_path)
        bus.data.import_catalog(input_path=catalog_path)

        (bus.data.catalog.s_in,
        bus.data.catalog.s_out,
        bus.data.catalog.l_sun,
        bus.data.catalog.hz_in,
        bus.data.catalog.hz_out,
        bus.data.catalog.hz_center) = single_habitable_zone(model='Kopparapu-Conservative',
                                                            temp_s=bus.data.catalog.temp_s,
                                                            radius_s=bus.data.catalog.radius_s)

        bus.data.catalog.habitable = np.logical_and.reduce((
            (bus.data.catalog['semimajor_p'] > bus.data.catalog['hz_in']).to_numpy(),
            (bus.data.catalog['semimajor_p'] < bus.data.catalog['hz_out']).to_numpy(),
            (bus.data.catalog['radius_p'].ge(0.8*(bus.data.catalog.semimajor_p/np.sqrt(bus.data.catalog.l_sun))**(-0.5))).to_numpy(),
            (bus.data.catalog['radius_p'].le(1.4)).to_numpy()))

        bus.data.catalog['is_interesting'] = False
        for exp in bus.data.options.optimization['experiments'].keys():
            mask_exp = ((bus.data.catalog.radius_p
                        >= bus.data.options.optimization['experiments'][exp]['radius_p_min'])
                        & (bus.data.catalog.radius_p
                        <= bus.data.options.optimization['experiments'][exp]['radius_p_max'])
                        & (bus.data.catalog.temp_s
                        >= bus.data.options.optimization['experiments'][exp]['temp_s_min'])
                        & (bus.data.catalog.temp_s
                        <= bus.data.options.optimization['experiments'][exp]['temp_s_max']))

            if bus.data.options.optimization['experiments'][exp]['in_HZ']:
                mask_exp = (mask_exp
                            & (bus.data.catalog['habitable']))

            bus.data.catalog['exp_' + exp] = mask_exp

            bus.data.catalog['is_interesting'] = np.logical_or(mask_exp, bus.data.catalog['is_interesting'])

        eta_all = np.sum(bus.data.catalog.habitable) / len(np.unique(bus.data.catalog.name_s)) / len(np.unique(bus.data.catalog.nuniverse))
        eta_exp = np.sum(bus.data.catalog.is_interesting) / len(np.unique(bus.data.catalog[bus.data.catalog.is_interesting].name_s)) / len(np.unique(bus.data.catalog.nuniverse))
        eta_fgk = np.sum(bus.data.catalog.habitable[bus.data.catalog.stype != 'M']) / len(np.unique(bus.data.catalog.name_s[bus.data.catalog.stype != 'M'])) / len(np.unique(bus.data.catalog.nuniverse))
        eta_m = np.sum(bus.data.catalog.habitable[bus.data.catalog.stype == 'M']) / len(np.unique(bus.data.catalog.name_s[bus.data.catalog.stype == 'M'])) / len(np.unique(bus.data.catalog.nuniverse))

        return float(eta_all), float(eta_exp), float(eta_fgk), float(eta_m)

    def get_all_etas(self, 
                     catalogs,
                     csv_path,
                     catalog_base_path,
                     catalog_name):
        etas_all = {}
        etas_exp = {}
        etas_fgk = {}
        etas_m = {}

        pbar = tqdm(catalogs)

        for catalog in pbar:
            pbar.set_description(f"Processing {catalog}")

            catalog_path = os.path.join(catalog_base_path, catalog, catalog_name)
            eta_all, eta_exp, eta_fgk, eta_m = self.get_eff_eta(catalog_path)
            etas_all[catalog] = eta_all
            etas_exp[catalog] = eta_exp
            etas_m[catalog] = eta_m
            etas_fgk[catalog] = eta_fgk

        # Combine into a single DataFrame
        etas = pd.DataFrame({
            'etas_exp': etas_exp,
            'etas_all': etas_all,
            'etas_fgk': etas_fgk,
            'etas_m': etas_m
        })

        # Optional: Reset the index if you want the names as a regular column
        # etas = etas.reset_index().rename(columns={'index': 'scenario'})

        etas.to_csv(csv_path)

    def run_eta_analysis(self, path):
        base = Path("/home/kirschkobold/documents/life_internship/LIFEsim_yields/euler_data")
        exclude = {"config_files", "logs"}
        catalogs = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in exclude)