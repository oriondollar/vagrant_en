import os
import logging

from qm9.data.prepare.qm9 import download_dataset_qm9

def prepare_dataset(datadir, dataset, calc_bonds, subset=None, splits=None, cleanup=False, force_download=False, reprocess=False):
    # If datasets have subsets,
    if subset:
        dataset_dir = [datadir, dataset, subset]
    else:
        dataset_dir = [datadir, dataset]

    # Names of splits, based upon keys if split dictionary exists, elsewise default to train/valid/test.
    split_names = splits.keys() if splits is not None else [
        'train', 'valid', 'test']

    # Assume one data file for each split
    datafiles = {split: os.path.join(
        *(dataset_dir + [split + '.npz'])) for split in split_names}

    # Check datafiles exist
    datafiles_checks = [os.path.exists(datafile)
                        for datafile in datafiles.values()]

    # Check if prepared dataset exists, and if not set flag to download below.
    # Probably should add more consistency checks, such as number of datapoints, etc...
    new_download = False
    if all(datafiles_checks):
        logging.info('Dataset exists and is processed.')
    elif all([not x for x in datafiles_checks]):
        # If checks are failed.
        new_download = True
    else:
        raise ValueError(
            'Dataset only partially processed. Try deleting {} and running again to download/process.'.format(os.path.join(dataset_dir)))

    # If need to download dataset, pass to appropriate downloader
    if new_download or force_download:
        logging.info('Dataset does not exist. Downloading!')
        if dataset.lower().startswith('qm9'):
            download_dataset_qm9(datadir, dataset, calc_bonds, splits, cleanup=cleanup)
        else:
            raise ValueError(
                'Repo currently only configured to work with QM9')
    elif reprocess:
        print('reprocessing data...')
        if dataset.lower().startswith('qm9'):
            download_dataset_qm9(datadir, dataset, calc_bonds, splits, cleanup=cleanup, download=False)
        else:
            raise ValueError(
                'Repo currently only configured to work with QM9')

    return datafiles
