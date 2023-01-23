import os, logging
from urllib.request import urlopen

def download_data(url, outfile='', binary=False):
    # Try statement to catch downloads.
    try:
        # Download url using urlopen
        with urlopen(url) as f:
            data = f.read()
        logging.info('Data download success!')
        success = True
    except:
        logging.info('Data download failed!')
        success = False

    if binary:
        # If data is binary, use 'wb' if outputting to file
        writeflag = 'wb'
    else:
        # If data is string, convert to string and use 'w' if outputting to file
        writeflag = 'w'
        data = data.decode('utf-8')

    if outfile:
        logging.info('Saving downloaded data to file: {}'.format(outfile))

        with open(outfile, writeflag) as f:
            f.write(data)

    return data, success

def is_int(str):
    try:
        int(str)
        return True
    except:
        return False

def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass
