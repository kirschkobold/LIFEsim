import lifesim

path = "/cluster/project/quanz/kseliner/yields/runs/20260429_merge"
config_path = "/cluster/project/quanz/kseliner/yields/runs/20260429_merge/config_files/optimizer_config.yaml"
save_path = "/cluster/project/quanz/kseliner/yields/runs/20260429_analysis"

ya = lifesim.YieldAnalysis(catalog_folder_path = '$catalog_folder_path', 
                           config_path = '$config_path',
                           option_name = '$option_fullname',
                           option_unit = '$option_unit',
                           save_path = '$save_path')


ya.run_interpolation()
ya.run_eta_summary()

ya.plot_all_single_opts()
ya.plot_all_one_cat()
ya.plot_all_final()