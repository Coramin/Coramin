import os
from zipfile import ZipFile
from pyomo.common import fileutils
from pyomo.common import download
import enum
import math
import csv


def get_instancedata(download_dir=None):
    """
    Downlaod instancedata.csv from MINLPLib

    Parameters
    ----------
    download_dir: str
        The directory in which to place the downloaded file. The default will be a 
        directory called minlplib in the current working directory.
    """
    if download_dir is None:
        download_dir = os.path.join(os.getcwd(), 'minlplib')

    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    downloader = download.FileDownloader()
    downloader.set_destination_filename(os.path.join(download_dir, 'instancedata.csv'))
    downloader.get_binary_file('http://www.minlplib.org/instancedata.csv')


def filter_cases(instancedata_download_dir=None,
                 min_nvars=0, max_nvars=math.inf,
                 min_nbinvars=0, max_nbinvars=math.inf,
                 min_nintvars=0, max_nintvars=math.inf,
                 min_nnlvars=0, max_nnlvars=math.inf,
                 min_nnlbinvars=0, max_nnlbinvars=math.inf,
                 min_nnlintvars=0, max_nnlintvars=math.inf,
                 formats=None,
                 probtype=None,
                 objtype=None,
                 objcurvature=None,
                 min_nobjnz=0, max_nobjnz=math.inf,
                 min_nobjnlnz=0, max_nobjnlnz=math.inf,
                 min_ncons=0, max_ncons=math.inf,
                 min_nlincons=0, max_nlincons=math.inf,
                 min_nquadcons=0, max_nquadcons=math.inf,
                 min_npolynomcons=0, max_npolynomcons=math.inf,
                 min_nsignomcons=0, max_nsignomcons=math.inf,
                 min_ngennlcons=0, max_ngennlcons=math.inf,
                 min_nloperands=0, max_nloperands=math.inf,
                 conscurvature=None,
                 min_njacobiannz=0, max_njacobiannz=math.inf,
                 min_njacobiannlnz=0, max_njacobiannlnz=math.inf,
                 min_nnz=0, max_nnz=math.inf,
                 min_nlaghessiannz=0, max_nlaghessiannz=math.inf,
                 min_nlaghessiandiagnz=0, max_nlaghessiandiagnz=math.inf,
                 min_nsemi=0, max_nsemi=math.inf,
                 min_nnlsemi=0, max_nnlsemi=math.inf,
                 min_nsos1=0, max_nsos1=math.inf,
                 min_nsos2=0, max_nsos2=math.inf,
                 c=None):
    if instancedata_download_dir is None:
        instancedata_download_dir = os.path.join(os.getcwd(), 'minlplib')

    filename = os.path.join(instancedata_download_dir, 'instancedata.csv')

    if not os.path.exists(instancedata_download_dir):
        os.mkdir(instancedata_download_dir)

    if not os.path.exists(filename):
        get_instancedata(download_dir=instancedata_download_dir)

    csv_file = open(filename, 'r')
    reader = csv.reader(csv_file, delimiter=';')
    headings = {column: ndx for ndx, column in enumerate(next(reader))}
    rows = [row for row in reader]
    csv_file.close()

    cases = list()
    for ndx, row in enumerate(rows):
        if formats is not None:
            available_formats = row[headings['formats']]
            available_formats = available_formats.replace('set([', '')
            available_formats = available_formats.replace('])', '')
            available_formats = available_formats.replace(' ', '')
            available_formats = available_formats.split(',')
            available_formats = set(available_formats)

            if formats not in available_formats:
                continue

        nvars = int(row[headings['nvars']])
        if nvars < min_nvars or nvars > max_nvars:
            continue

        nbinvars = int(row[headings['nbinvars']])
        if nbinvars < min_nbinvars or nbinvars > max_nbinvars:
            continue

        cases.append(row[headings['name']])

    return cases


def get_minlplib(download_dir=None, format='osil', cases=None):
    """
    Download MINLPLib

    Parameters
    ----------
    download_dir: str
        The directory in which to place the downloaded files. The default will be a 
        current_working_directory/minlplib/file_format/.
    format: str
        The file format requested. Options are ams, gms, lp, mod, nl, osil, and pip
    conditions: dict
        Conditions to place on the downloaded test problems.
    """
    pass
