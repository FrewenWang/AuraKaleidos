import argparse
import json
import os
import shutil
import urllib.request
import zipfile
import sys

class GetAura2:
    def __init__(self, config_json):
        with open(config_json, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def download(self, lib):
        info = lib.split('_')

        if len(info) < 3:
            raise ValueError(f'{lib} is not a valid library name')

        url = f'https://FDS_REGION-fds.api.xiaomi.net/FDS_BUCKET_NAME/{info[1]}/{lib}.zip'
        total_size = -1
        downloaded = 0
        chunk_size = 8192

        print("Start to download dependencies: ", lib)

        with urllib.request.urlopen(url) as response:
            if 'content-length' in response.info():
                total_size = int(response.info()['content-length'])
            if response.status == 200:
                with open(f'{lib}.zip', 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = downloaded / total_size * 100
                            print(f"download progress: {progress:.2f}% ({downloaded} / {total_size} bytes)", end='\r', file=sys.stdout)

                if os.path.exists(lib):
                    shutil.rmtree(lib)

                with zipfile.ZipFile(f'{lib}.zip', 'r') as z:
                    z.extractall()

                os.remove(f'{lib}.zip')
            else:
                print(f'download {lib} failed')

    def update(self, libs):
        if libs is None or libs == 'all':
            for lib, enable in self.config.items():
                if enable:
                    self.download(lib)
        else:
            for lib in set(libs.replace(' ', '').split(',')):
                if lib in self.config and self.config[lib] == 1:
                    self.download(lib)
                else:
                    print(f'library {lib} is not configured')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get/Update Aura2')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--directory', default=script_dir, type=str,
                        help='directory for downloading libraries')
    parser.add_argument('--update', default=None, type=str,
                        help=('if all, update all enabled libraries in the configuration file;'
                              'otherwise, update the specified libraries (separated by commas)'))
    parser.add_argument('config', type=str,
                        help='json configuration file, containing the libraries you may need')

    args = parser.parse_args()

    aura = GetAura2(args.config)

    if not (os.path.exists(args.directory) and os.path.isdir(args.directory)):
        raise ValueError(f'{args.directory} is not a directory')

    os.chdir(args.directory)

    aura.update(args.update)
