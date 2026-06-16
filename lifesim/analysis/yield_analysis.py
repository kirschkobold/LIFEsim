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
from lifesim.analysis.yield_wrapper import extract_float_from_name

class YieldAnalysis:
    def __init__(self,
                 catalog_folder_path,
                 config_path,
                 option_name,
                 option_unit,
                 save_path=None):
        self.catalog_folder_path = catalog_folder_path
        self.config_path = config_path
        self.option_name = option_name
        self.option_unit = option_unit
        self.save_path = Path(save_path) if save_path else None
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)

    def interpolate_one(self, 
                        source_path,
                        kind_option=None):
        print(f"Processing {source_path}")
        subdirs = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
        diams_float = [extract_float_from_name(d) for d in subdirs]
        diams = [str(value).replace('.', '_') for value in diams_float]

        # uu = upper uncertainty, lu = lower uncertainty
        mission_time = pd.DataFrame(index=np.sort(diams_float), columns=['mission_time_mean', 'lu_mission_time', 'uu_mission_time', 'mission_time_opt_factor'])

        for subdir, d in zip(subdirs, diams_float):
            data_path = os.path.join(source_path, subdir)
            # read the csv file in data path that ends with _mission_time.csv, return error if multiple or none found
            csv_files = [f for f in os.listdir(data_path) if f.endswith('_mission_time.csv')]
            if len(csv_files) != 1:
                raise ValueError(f'Expected one _mission_time.csv file in {data_path}, found {len(csv_files)}')
            data = pd.read_csv(os.path.join(data_path, csv_files[0]))
            data_finite = data[np.isfinite(data['total'])] # remove any possible inf rows
            rows_removed = len(data) - len(data_finite)
            print(f"in {subdir}: Removed {rows_removed} row(s) with non-finite values in 'total' column")
            mission_time.loc[d, 'mission_time_mean'] = data_finite['total'].mean()
            mission_time.loc[d, 'lu_mission_time'] = data_finite['total'].mean() - data_finite['total'].quantile(0.16)
            mission_time.loc[d, 'uu_mission_time'] = data_finite['total'].quantile(0.84) - data_finite['total'].mean()
            det_ratio = data_finite['detection'] / (data_finite['total'])
            mission_time.loc[d, 'detection_ratio_mean'] = det_ratio.mean()
            mission_time.loc[d, 'lu_detection_ratio'] = det_ratio.mean() - det_ratio.quantile(0.16)
            mission_time.loc[d, 'uu_detection_ratio'] = det_ratio.quantile(0.84) - det_ratio.mean()

            if 'n_Experiment_1' in data.columns:
                mission_time.loc[d, 'mission_time_opt_factor'] = data_finite.loc[data_finite['n_Experiment_1'] == data_finite['n_Experiment_1'].max(), 'total'].mean()
            else:
                mission_time.loc[d, 'mission_time_opt_factor'] = data_finite.loc[data_finite['n_Experiment_2'] == data_finite['n_Experiment_2'].max(), 'total'].mean()

        # add jitter for possible duplicate values
        def add_jitter_to_duplicates(series):
            vals = series.values.copy().astype(float)
            seen = {}
            for i, v in enumerate(vals):
                if v in seen:
                    seen[v] += 1
                    vals[i] += seen[v] * 1e-3  
                else:
                    seen[v] = 0
            n_duplicates = sum(v for v in seen.values())
            if n_duplicates > 0:
                print(f"Jitter applied to {n_duplicates} duplicate value(s) in '{series.name}'")
            return vals

        mission_time['mission_time_mean'] = add_jitter_to_duplicates(mission_time['mission_time_mean'])
        mission_time['mission_time_opt_factor'] = add_jitter_to_duplicates(mission_time['mission_time_opt_factor'])

        max_times = np.arange(0.5, 20.1, 0.5)
        
        # check the overlap for extrapolation
        target_min, target_max = max_times.min(), max_times.max()
        mission_times_years = np.array(mission_time['mission_time_mean'], dtype=float) / 365.25 / 24 / 3600
        data_min, data_max = mission_times_years.min(), mission_times_years.max()
        overlap_min = max(data_min, target_min)
        overlap_max = min(data_max, target_max)
        overlap_range = target_max - target_min
        
        if overlap_max <= overlap_min:
            print(f"WARNING: No input data overlaps the interpolation range [{target_min}, {target_max}] years.")
            print(f"Input data lies in [{data_min:.2f}, {data_max:.2f}] years.")
            print(f"Output is pure extrapolation and may be unreliable.")
        else:
            overlap_fraction = (overlap_max - overlap_min) / overlap_range
            print(f"INFO: Overlap fraction of input data with interpolation range is {overlap_fraction:.0%}, rest is extrapolation.")

        max_time_table = pd.DataFrame(index=np.sort(max_times), columns=['diameter_mean', 'lu_diameter', 'uu_diameter', 'diameter_opt_factor'])

        n = len(mission_time)
        kind = kind_option if kind_option is not None else 'cubic' if n >= 4 else 'quadratic' if n >= 3 else 'linear'
        print(f"INFO: Using {kind} interpolation for {n} data points.")

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

    def run_interpolation(self, kind_option=None):
        # run the interpolation on all opt_ directories in the given path

        # find all subdirectories (and subsub, and so on) in path that start with 'opt_'
        opt_dirs = []
        for root, dirs, files in os.walk(self.catalog_folder_path):
            for dirname in dirs:
                if dirname.startswith('opt_'):
                    opt_dirs.append(os.path.join(root, dirname))

        for opt_dir in opt_dirs:
            self.interpolate_one(opt_dir, kind_option=kind_option)

    def get_eff_eta(self, 
                    catalog_path):
        bus = lifesim.Bus()

        # setting the options
        bus.build_from_config(filename=self.config_path)
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

    def run_eta_summary(self):
        base = Path(self.catalog_folder_path)
        exclude = {"config_files", "logs"}
        catalogs = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in exclude)

        csv_path = os.path.join(self.catalog_folder_path, f"eta_summary.csv")
        options = sorted(p.name for p in (Path(self.catalog_folder_path) / catalogs[0] / "output/ap_merged").iterdir() if p.is_dir())
        catalog_path = f"output/ap_merged/{options[0]}/sweep_{options[0]}_catalog.hdf5"
        
        self.get_all_etas(catalogs=catalogs,
                          csv_path=csv_path,
                          catalog_base_path=self.catalog_folder_path,
                          catalog_name=catalog_path)
    
    def plot_single_opt(self, 
                        path,
                        text_vertical=False,
                        legend=True):
        fig, ax = plt.subplots(dpi=300, figsize=(5,5))

        name = [x for x in path.split('/') if x != ''][-1]
        mission_time = pd.read_csv(os.path.join(path, name + '_diameter_mission_time_RAW.csv'), index_col=0)
        max_time_table = pd.read_csv(os.path.join(path, name + '_diameter_mission_time.csv'), index_col=0)

        times_select = [5, 10]

        ax.plot(mission_time.index, mission_time['mission_time_opt_factor'] / 365.25 / 24 / 60 / 60, label='Opt. Factor Mission Time', linestyle='--', c='tab:orange', marker='x')
        ax.plot(mission_time.index, mission_time['mission_time_mean'] / 365.25 / 24 / 60 / 60, label='Mean Mission Time', c='k', marker='x')
        ax.fill_between(list(mission_time.index),
                        list((mission_time['mission_time_mean'] - mission_time['lu_mission_time']) / 365.25 / 24 / 60 / 60),
                        list((mission_time['mission_time_mean'] + mission_time['uu_mission_time']) / 365.25 / 24 / 60 / 60),
                        color='gray', alpha=0.5, label='68% Confidence Interval')
        ax.set_xlabel(f'{self.option_name} ({self.option_unit})')
        ax.set_ylabel('Mission Time (years)')

        catalog_name = path.split('/')[-3]
        ax.set_title(f'{catalog_name}\n{name}')

        if legend:
            ax.legend()

        # ax.xscale('log')
        ax.set_yscale('log')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        for t in times_select:
            ax.hlines(y=t, xmin=xlim[0], xmax=max_time_table['diameter_mean'].loc[t],  color='k', linestyle=':')
            if text_vertical:
                ax.text(x=1.0*max_time_table['diameter_mean'].loc[t], y=1.2*t, s=f'{t} years', verticalalignment='center', horizontalalignment='left', rotation=90, rotation_mode='anchor')
            else:
                ax.text(x=1.07*max_time_table['diameter_mean'].loc[t], y=t, s=f'{t} years', verticalalignment='center', horizontalalignment='left')
            ax.vlines(x=max_time_table['diameter_mean'].loc[t], ymin=0, ymax=t, color='k', linestyle=':')
            # check for negative error bars values
            lu = max_time_table['lu_diameter'].loc[t]
            uu = max_time_table['uu_diameter'].loc[t]
            if lu >= 0 and uu >= 0:
                ax.errorbar(x=max_time_table['diameter_mean'].loc[t],
                            y=t,
                            xerr=[[max_time_table['lu_diameter'].loc[t]], [max_time_table['uu_diameter'].loc[t]]],
                            fmt='s',
                            color='k',
                            label=f'{self.option_name} at {t} years' if t == times_select[0] else None)
            else:
                print(f"Skipping xerr for t={t}: negative uncertainty values (lu={lu}, uu={uu})")
                ax.plot(max_time_table['diameter_mean'].loc[t],
                        t,
                        's',
                        color='k',
                        label=f'{self.option_name} at {t} years' if t == times_select[0] else None)

        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        if self.save_path:
            subfolder = self.save_path / "single_opt_plots"
            subfolder.mkdir(parents=True, exist_ok=True)
            plot_name = Path(path).parts[-3] + "_" + Path(path).name + ".png"
            plt.savefig(subfolder / plot_name, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_all_single_opts(self):
        base = Path(self.catalog_folder_path)
        exclude = {"config_files", "logs"}
        catalogs = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in exclude)

        experiments_path = Path(self.catalog_folder_path) / catalogs[0] / "output"
        exclude = {"ap_merged"}
        experiments = sorted(p.name for p in experiments_path.iterdir() if p.is_dir() and p.name not in exclude)

        for catalog in catalogs:
            print(f"Processing catalog {catalog}...")
            for experiment in experiments:
                experiment_path = Path(self.catalog_folder_path) / catalog / "output" / experiment
                print(f"Processing {experiment}...")
                self.plot_single_opt(path=str(experiment_path))

    def _get_catalog(self,
                     source_path):
        subdirs = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d)) if d.startswith('opt_')]
        opt_names = [d.removeprefix('opt_') for d in subdirs]

        diam_time_tables = {}
        for subdir, opt_name in zip(subdirs, opt_names):
            data_path = os.path.join(source_path, subdir)
            diam_time_tables[opt_name] = pd.read_csv(os.path.join(data_path, subdir + '_diameter_mission_time.csv'), index_col=0)
            diam_time_tables[opt_name][diam_time_tables[opt_name] < 0] = np.nan  # set negative values to nan

        # sort opt_name by mean diameter at 10 years
        diam_time_tables = dict(sorted(diam_time_tables.items(), key=lambda item: item[1].loc[10, 'diameter_mean']))

        return diam_time_tables

    def one_catalog_all_opt_plot(self, 
                                 cat_name):
        source_path = Path(self.catalog_folder_path) / cat_name / 'output'
        catalog_time_tables = self._get_catalog(source_path)

        configs = {'e1_e2_f05_char': dict(label='Exp. 1 & 2; Char.', color='tab:orange', ls='-',   z=5),
                   'e1_e2_f09_char': dict(label='Exp. 1 & 2; Char.', color='tab:orange', ls='-',   z=5),
                   'e1_e2_f05':      dict(label='Exp. 1 & 2; Full',  color='gray',       ls='--',  z=4),
                   'e1_e2_f09':      dict(label='Exp. 1 & 2; Full',  color='gray',       ls='--',  z=4),
                   'e1_f05_char':    dict(label='Exp. 1; Char.',      color='gray',       ls='-.',  z=3),
                   'e1_f09_char':    dict(label='Exp. 1; Char.',      color='gray',       ls='-.',  z=3)}

        fig, ax = plt.subplots(dpi=200, ncols=2, figsize=(8, 4))

        for opt, cfg in configs.items():
            axis = ax[0] if 'f05' in opt else ax[1]
            axis.plot(catalog_time_tables[opt].index, 
                      catalog_time_tables[opt]['diameter_mean'], 
                      label=cfg['label'], 
                      color=cfg['color'], 
                      linestyle=cfg['ls'], 
                      zorder=cfg['z'])
            axis.set_xlabel('Mission Time (years)')
            axis.set_ylabel(f'{self.option_name} ({self.option_unit})')
            axis.grid(True, which='both', linestyle='-', linewidth=0.5)
            axis.set_title(f'{cat_name}\n- to {"50%" if "f05" in opt else "90%"}', fontsize=10)

        ax[1].legend()

        # yaxis of right plot to the right
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()

        # adjust to common y-axis
        ymin = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0])
        ymax = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
        ax[0].set_ylim(ymin, ymax)
        ax[1].set_ylim(ymin, ymax)

        if self.save_path:
            subfolder = self.save_path / "one_cat_all_opt_plots"
            subfolder.mkdir(parents=True, exist_ok=True)
            plot_name = cat_name + ".png"  # Path(path).parts[-3] + "_" + Path(path).name + ".png"
            plt.savefig(subfolder / plot_name, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_all_one_cat(self):
        base = Path(self.catalog_folder_path)
        exclude = {"config_files", "logs"}
        catalogs = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in exclude)

        for catalog in catalogs: 
            print(f"Processing catalog {catalog}...")
            self.one_catalog_all_opt_plot(cat_name=catalog)

    def separate_bryson_sag(self,
                            opt,
                            mtime):
        val_bryson = []
        val_sag = []

        etas = pd.read_csv(Path(self.catalog_folder_path) / f"eta_summary.csv", index_col=0)

        for catalog in etas.index:
            source_path = Path(self.catalog_folder_path) / catalog / 'output'
            catalog_time_tables = self._get_catalog(source_path)
            if 'Bryson' in catalog:
                val_bryson.append([etas.loc[catalog, 'etas_fgk'], catalog_time_tables[opt]['diameter_mean'].loc[mtime]])
            elif 'SAG' in catalog:
                val_sag.append([etas.loc[catalog, 'etas_fgk'], catalog_time_tables[opt]['diameter_mean'].loc[mtime]])
        
        val_bryson = np.array(val_bryson)
        val_bryson = val_bryson[val_bryson[:, 0].argsort()]  # sort by eta
        val_sag = np.array(val_sag)
        val_sag = val_sag[val_sag[:, 0].argsort()]  # sort by eta

        return val_bryson, val_sag
    
    def plot_final_one(self,
                       opt):
        fig, ax = plt.subplots(dpi=200, ncols=2, figsize=(8, 4), gridspec_kw={'width_ratios': [4, 3]})

        exp_str = 'Exp. ' + ' and '.join(p[1:] for p in opt.split('_') if p.startswith('e') and p[1:].isdigit())
        char_str = 'char. opt.' if 'char' in opt.split('_') else 'not char. opt.'
        fig.suptitle(f'{exp_str}, {char_str}', fontsize=12, x=0.9, y=0.8, ha='right', va='top')

        val_bryson, val_sag = self.separate_bryson_sag(opt=opt, mtime=5)
        ax[0].plot(val_sag[:, 0], val_sag[:, 1], marker='x', linestyle='-', label='SxD', color='tab:orange')
        ax[0].plot(val_bryson[:, 0], val_bryson[:, 1], marker='x', linestyle='--', label='BxD', color='tab:orange')

        opt2 = opt.replace('f09', 'f05')
        val_bryson, val_sag = self.separate_bryson_sag(opt=opt2, mtime=5)
        ax[0].plot(val_sag[:, 0], val_sag[:, 1], marker='x', linestyle='-', label='SxD', color='gray')
        ax[0].plot(val_bryson[:, 0], val_bryson[:, 1], marker='x', linestyle='--', label='BxD', color='gray')

        ax2 = ax[0].twiny()
        ax2.set_xticks([])

        etas = pd.read_csv(Path(self.catalog_folder_path) / f"eta_summary.csv", index_col=0)
        etas['shortname'] = etas.index.str.split('_').str[1].str.replace(r'([A-Z])[^x]*', r'\1', regex=True) + '\n' + etas.index.str.split('_').str[2].str[:4]
        marker_labels = etas['shortname'].to_list()
        marker_positions = etas['etas_fgk'].to_list()
        ax2.set_xticks(marker_positions)
        ax2.set_xticklabels(marker_labels, rotation=45)
        ax2.set_xlim(ax[0].get_xlim())

        handles = [
            plt.Line2D([0], [0], color='k', linestyle='-', label='SAG13 x Dressing'),
            plt.Line2D([0], [0], color='k', linestyle='--', label='Bryson x Dressing'),
            plt.Line2D([0], [0], marker='x', color='tab:orange', linestyle='', label='to 90%'),
            plt.Line2D([0], [0], marker='x', color='gray', linestyle='', label='to 50%'),
        ]
        ax[0].legend(handles=handles, loc='upper right', title=f'{exp_str}, {char_str}, in 5 years')

        for i, col in zip([5, 10, 15], ['lightgrey', 'gray', 'tab:orange']):
            val_bryson, val_sag = self.separate_bryson_sag(opt=opt, mtime=float(i))
            ax[1].plot(val_sag[:, 0], val_sag[:, 1], marker='x', linestyle='-', label='SxD', color=col)
            ax[1].plot(val_bryson[:, 0], val_bryson[:, 1], marker='x', linestyle='--', label='BxD', color=col)
        
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()

        handles = [
            plt.Line2D([0], [0], marker='x', color='tab:orange', linestyle='', label='in 5 yrs'),
            plt.Line2D([0], [0], marker='x', color='gray', linestyle='', label='in 10 yrs'),
            plt.Line2D([0], [0], marker='x', color='lightgray', linestyle='', label='in 15 yrs'),
        ]
        ax[1].legend(handles=handles, loc='upper right')

        for i in range(2):
            ax[i].set_xlabel(r'$\eta_\mathrm{Earth, FGK}$; EEC Ratio')
            ax[i].set_ylabel(f'Required {self.option_name} ({self.option_unit})')
            ax[i].grid(True, which='both', linestyle='-', linewidth=0.5)

        ymin = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0])
        ymax = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
        ax[0].set_ylim(ymin, ymax)
        ax[1].set_ylim(ymin, ymax)

        exp_str = 'Exp. ' + ' and '.join(p[1:] for p in opt.split('_') if p.startswith('e') and p[1:].isdigit())
        char_str = 'char. opt.' if 'char' in opt.split('_') else 'not char. opt.'
        fig.suptitle(f'{exp_str}, {char_str}', fontsize=12, x=0.9, y=0.8, ha='right', va='top')

        fig.tight_layout()

        if self.save_path:
            subfolder = self.save_path / "final_plots"
            subfolder.mkdir(parents=True, exist_ok=True)
            plot_name = opt + ".png"  # Path(path).parts[-3] + "_" + Path(path).name + ".png"
            plt.savefig(subfolder / plot_name, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_all_final(self):
        base = Path(self.catalog_folder_path)
        exclude = {"config_files", "logs"}
        catalogs = sorted(p.name for p in base.iterdir() if p.is_dir() and p.name not in exclude)

        experiments_path = Path(self.catalog_folder_path) / catalogs[0] / "output"
        exclude = {"ap_merged"}
        experiments = sorted(p.name for p in experiments_path.iterdir() if p.is_dir() and p.name not in exclude and 'f09' in p.name)
        experiments_clean = [s.removeprefix('opt_') for s in experiments]

        for experiment in experiments_clean:
            print(f"Processing {experiment}...")
            self.plot_final_one(opt=experiment)
        