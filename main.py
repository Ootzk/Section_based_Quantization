import os
from datetime import datetime
import shutil
import sys
from pathlib import Path
import json

from architectures.train import run


save_architectures = True


def save_current_component_and_model_files_and_given_configuration_file(config_as_argument):
    with open(config_as_argument, 'r') as config_file:
        config = json.load(config_file)
        
        title = config['experiment']['title']
        ID = str(config['experiment']['ID']).zfill(2)
        
        os.makedirs('experiments/{0}/{1}'.format(title, ID))
        config['datetime'] = datetime.now().strftime("%Y%m%d")
        with open('experiments/{0}/{1}/config.json'.format(title, ID), 'w') as save_config_file:
            json.dump(config, save_config_file, indent=4)
    
    if save_architectures:
        shutil.copytree('architectures',
                        'experiments/{0}/{1}/architectures'.format(title, ID))
    return config
        
    
        
if __name__ == "__main__":
    if os.path.isdir(sys.argv[1]):
        for filename in sorted(os.listdir(sys.argv[1])):
            if filename.endswith('.json'):
                config = save_current_component_and_model_files_and_given_configuration_file('Z/{0}'.format(filename))
                run(config)
    else:
        config = save_current_component_and_model_files_and_given_configuration_file(sys.argv[1])
        run(config)