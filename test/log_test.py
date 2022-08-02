import logging
import logging.config
import yaml

def parse_config(config_file_path: str) -> dict:
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    # change something manually
    return config

if __name__ == '__main__':
    config = parse_config('config.yaml')

    print(config['tol']+1)

    '''logging.config.dictConfig(config['logger'])
    logger = logging.getLogger()
    logger.info(config)
    logger.handlers[1].name

    print(logger.handlers[0])'''

