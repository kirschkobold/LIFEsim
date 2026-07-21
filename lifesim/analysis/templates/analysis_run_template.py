import lifesim

ya = lifesim.YieldAnalysis(catalog_folder_path = '$catalog_folder_path', 
                           config_path = '$config_path',
                           option_name = '$option_fullname',
                           option_unit = '$option_unit',
                           save_path = '$save_path')


ya.run_interpolation(kind_option='$kind_option')
ya.run_eta_summary()

ya.plot_all_single_opts()
ya.plot_all_one_cat()
ya.plot_all_final(catalog_mode = '$catalog_mode')
ya.final_plots_data(catalog_mode = '$catalog_mode')
ya.plot_final_fit(catalog_mode = '$catalog_mode')