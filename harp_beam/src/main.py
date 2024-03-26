import configparser

# initialise the config parser
config = configparser.ConfigParser()

# read the config file
config.read('config.ini')

# accessing variables
num_dir = config.getfloat('BEAM', 'num_dir')
freq = config.getfloat('BEAM', 'freq')
c0 = config.getfloat('BEAM', 'speed_of_light')
antenna_name = config.get('ANETENNA', 'antenna_name')
array_layout = config.get('ANETENNA', 'array_layout')
data_folder = config['PATHS']['data_folder']
filename_eep = config['PATHS']['filename_eep']
filename_eep = filename_eep % {'antenna_name': antenna_name, 'array_layout': array_layout, 'freq': freq}
