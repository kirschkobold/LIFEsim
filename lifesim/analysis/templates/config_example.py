# personal configurations
yields_path = "/cluster/project/quanz/YOUR_USERNAME/yields"           # path to the yields folder (with subfolders catalogs and runs)
venv_path = "/cluster/home/YOUR_USERNAME/LIFEsim/venv/bin/activate"   # path to the activate script of your virtual environment

# per run configurations
today = "20260429"                 # format: YYYYMMDD
catalog_source_date = "20260421"   # format: YYYYMMDD, date of the imported catalogs
catalog_mode = "default"                  # either "default" or "scaledetas"
queue = "public"                   # alternatively es_quanz

# sweep options
option_name =   "primary_temp"        # name of the option to sweep over
option_values = "np.arange(44,50)"    # values to sweep over
run_name =      "primarymirror_sweep" # name of the run, used for naming folders and files
option_fullname = "Primary Mirror Temperature" # full name of the option, used for plot labels
option_unit =   "K"                   # unit of the option, used for plot labels

# analysis options
kind_option = None # interpolation method to use, e.g. 'linear', 'quadratic', 'cubic' (if None, will be chosen based on number of data points)

# optional LIFEsim and optimizer configuration (leave as none if not using default template)
lifesim_config_path = None # eg. "/cluster/home/YOUR_USERNAME/LIFEsim/custom_config.yaml"
# important: must be utf-8 encoded! 
# use lifesim-yieldanalysis lifesim_config to generate a template
catalog_merge_path = None # eg. "/cluster/home/YOUR_USERNAME/LIFEsim/catalog_merge.csv" 
# important: catalog names must fit the pattern of the template
# use lifesim-yieldanalysis catalog_merge to generate a template
optimizer_scenarios_path = None # eg. "/cluster/home/YOUR_USERNAME/LIFEsim/optimizer_scenarios.csv" 
# use lifesim-yieldanalysis optimizer_scenarios to generate a template


