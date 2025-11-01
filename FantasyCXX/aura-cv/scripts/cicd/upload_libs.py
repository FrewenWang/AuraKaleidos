import logging
import os
import argparse

from fds import FDSClientConfiguration, GalaxyFDSClient

class FDSUpload:
    def __init__(self, region, access_key, secret_key, logger):
        self._endpoint   = '%s-fds.api.xiaomi.net' % region
        self._logger     = logger
        self._logger.info('endpoint: %s' % self._endpoint)

        self._fds_config = FDSClientConfiguration(region_name             = region,
                                                  enable_cdn_for_download = False,
                                                  enable_cdn_for_upload   = False,
                                                  enable_https            = True,
                                                  endpoint                = self._endpoint)

        self._fds_client = GalaxyFDSClient(access_key    = access_key,
                                           access_secret = secret_key,
                                           config        = self._fds_config)

    def put_object(self, data_file, bucket_name, object_name):
        # The object name doesn't have starting '/'
        object_name = object_name.lstrip('/')

        result_put  = None

        if data_file:
            with open(data_file, "rb") as f:
                result_put = self._fds_client.put_object(bucket_name, object_name, f)  # upload the object

                self._fds_client.set_public(bucket_name, object_name)  # set the object to be readable for all users
        else:
            self._logger.error('data_file is None')

        if result_put:
            self._logger.info('Put object %s success' % object_name)
        else:
            self._logger.error('Put object %s failed' % object_name)

    def put_directory(self, data_dir, bucket_name, object_name_prefix):
        object_name_prefix = object_name_prefix.lstrip('/')

        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                data_file = os.path.join(root, filename)

                if filename.startswith('.'):
                    self._logger.warn("object name can't start with '.', skipping: %s " % data_file)
                    continue
                if '/' in filename or '\\' in filename:
                    self._logger.warn("object name can't contain '/' or '\\', skipping: %s " % data_file)
                    continue

                object_name = os.path.normpath(os.path.join(object_name_prefix,
                                                            filename))
                object_name = '/'.join(object_name.split('\\'))

                self._logger.info('putting %s to %s/%s' % (data_file, bucket_name, object_name))

                self.put_object(data_file = data_file, bucket_name = bucket_name, object_name = object_name)

        self._logger.info('Upload directory success: %s, %s, %s' % (data_dir, bucket_name, object_name_prefix))
 

def main():
    parser = argparse.ArgumentParser(description='fds upload')

    parser.add_argument('-r', '--region',             type = str, required = True, help = 'region name')
    parser.add_argument('-k', '--access-key',         type = str, required = True, help = 'access key')
    parser.add_argument('-s', '--secret-key',         type = str, required = True, help = 'secret key')
    parser.add_argument('-b', '--bucket',             type = str, required = True, help = 'bucket name')

    parser.add_argument('-d', '--data-dir',           type = str, required = True, help = 'data dir for upload files')
    parser.add_argument('-p', '--object-name-prefix', type = str, required = True, help = 'object name prefix')

    args = parser.parse_args()

    log_format = '%(asctime)-15s [%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(level = logging.INFO, format = log_format)
    logger = logging.getLogger('fds.upload')

    fds_upload = FDSUpload(region = args.region, access_key = args.access_key, secret_key = args.secret_key, logger = logger)

    fds_upload.put_directory(data_dir = args.data_dir, bucket_name = args.bucket, object_name_prefix = args.object_name_prefix)

if __name__ == "__main__":
    """Upload the files of speicified directory to FDS.

    Args:
        region (str): region name.
        access_key (str):         The access key of the specified user.
        secret_key (str):         The secret key of the specified user.
        bucket (str):             The bucket name.
        data_dir (str):           The absolute dirctroy of upload files.
        object_name_prefix (str): The relative path of the object based on the bucket.

    Returns:
        None

    Usage:
        python3 upload_libs.py -r region -k access_key -s secret_key -b bucket -d data_dir -p object_name_prefix
    """
    print('**** FDS upload start ****')

    main()

    print('**** FDS upload end ****')
