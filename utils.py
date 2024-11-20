import os
import yaml

def get_apikey():
    """
    Reads API key from a configuration file.

    This function opens a configuration file named "apikeys.yml", reads the API key for OpenAI

    Returns:
        api_key (str): The OpenAI API key.
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "apikeys.yml")

    try:
        with open(file_path, 'r') as yamlfile:
            loaded_yamlfile = yaml.safe_load(yamlfile)
            API_KEY = loaded_yamlfile['openai']['api_key']
        return API_KEY
    except FileNotFoundError:
        print("Error: 'apikeys.yml' file not found.")
        raise
    except KeyError:
        print("Error: 'api_key' not found in 'openai' section.")
        raise
