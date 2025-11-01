import json
import argparse
import logging
import yaml

def parse_json_to_get_yml(json_file_path, start_key, end_key, yml_save_path, logger):
    """
    Parses a JSON file to extract the relative path of a specific key.
    """

    logger.info(f'Parsing JSON file: {json_file_path}, key: {start_key}, end key: {end_key}, yml file path: {yml_save_path}')

    # Load the JSON data from the file
    json_data = None
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    try:
        description = json_data['description']

        # Search Start and End key in the description
        start_index = description.find(start_key)
        end_index = description.find(end_key)
        
        if start_index == -1 or end_index == -1:
            logger.error(f'Configuration markers not found in the description of JSON file: {json_file_path}')
        else:
            config_text = description[start_index + len(start_key):end_index].strip()
            logger.info("\nExtracted Configuration Text:")
            logger.info(config_text)
            
            # Parse the YAML data
            try:
                parsed_data = yaml.safe_load(config_text)
                logger.info("\nParsed YAML data:")
                logger.info(parsed_data)

                # Save the parsed data to a YAML file
                with open(yml_save_path, 'w') as file:
                    yaml.dump(parsed_data, file, default_flow_style=False)

            except yaml.YAMLError as e:
                logger.error(f"Failed to parse YAML: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def main():
    """
    Main function that parses command-line arguments, finds the relative path
    from the JSON file, and downloads the corresponding file from GitLab.
    """
    parser = argparse.ArgumentParser(description = 'parse the file and save yml')

    parser.add_argument('-f',  '--json_file', type = str, required = True, help = 'json file path')
    parser.add_argument('-ys', '--yml_save',  type = str, required = True, help = 'the path of the yml file to save')

    log_format = '%(asctime)-15s [%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_format)
    logger = logging.getLogger('download_file')

    args = parser.parse_args()

    start_key = '### STRESS CONFIGURATION START ###'
    end_key   = '### STRESS CONFIGURATION END ###'

    parse_json_to_get_yml(args.json_file, start_key, end_key, args.yml_save, logger)

    logger.info('Parse yml and save yml file successfully')

if __name__ == '__main__':
    main()
