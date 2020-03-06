import os
from zipfile import ZipFile
from pyomo.common import fileutils
from pyomo.common import download
import enum
import math
import csv
from collections.abc import Iterable
import logging


logger = logging.getLogger(__name__)


def get_minlplib_instancedata(download_dir=None):
    """
    Downlaod instancedata.csv from MINLPLib which can be used to get statistics on the problems from minlplib.

    Parameters
    ----------
    download_dir: str
        The directory in which to place the downloaded file. The default will be a 
        directory called minlplib in the current working directory.
    """
    if download_dir is None:
        download_dir = os.path.join(os.getcwd(), 'minlplib')

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    downloader = download.FileDownloader()
    downloader.set_destination_filename(os.path.join(download_dir, 'instancedata.csv'))
    downloader.get_binary_file('http://www.minlplib.org/instancedata.csv')


def filter_minlplib_instances(instancedata_download_dir=None,
                              min_nvars=0, max_nvars=math.inf,
                              min_nbinvars=0, max_nbinvars=math.inf,
                              min_nintvars=0, max_nintvars=math.inf,
                              min_nnlvars=0, max_nnlvars=math.inf,
                              min_nnlbinvars=0, max_nnlbinvars=math.inf,
                              min_nnlintvars=0, max_nnlintvars=math.inf,
                              min_nobjnz=0, max_nobjnz=math.inf,
                              min_nobjnlnz=0, max_nobjnlnz=math.inf,
                              min_ncons=0, max_ncons=math.inf,
                              min_nlincons=0, max_nlincons=math.inf,
                              min_nquadcons=0, max_nquadcons=math.inf,
                              min_npolynomcons=0, max_npolynomcons=math.inf,
                              min_nsignomcons=0, max_nsignomcons=math.inf,
                              min_ngennlcons=0, max_ngennlcons=math.inf,
                              min_njacobiannz=0, max_njacobiannz=math.inf,
                              min_njacobiannlnz=0, max_njacobiannlnz=math.inf,
                              min_nlaghessiannz=0, max_nlaghessiannz=math.inf,
                              min_nlaghessiandiagnz=0, max_nlaghessiandiagnz=math.inf,
                              min_nsemi=0, max_nsemi=math.inf,
                              min_nnlsemi=0, max_nnlsemi=math.inf,
                              min_nsos1=0, max_nsos1=math.inf,
                              min_nsos2=0, max_nsos2=math.inf,
                              acceptable_formats=None,
                              acceptable_probtype=None,
                              acceptable_objtype=None,
                              acceptable_objcurvature=None,
                              acceptable_conscurvature=None,
                              acceptable_convex=None):
    """
    This function filters problems from MINLPLib based on
    instancedata.csv from MINLPLib and the conditions specified
    through the function arguments. The function argument names
    correspond to column headings from instancedata.csv. The 
    arguments starting with min or max require integer inputs. 
    The arguments starting with acceptable require either a 
    string or an iterable of strings. See the MINLPLib documentation 
    for acceptable values.
    """
    if instancedata_download_dir is None:
        instancedata_download_dir = os.path.join(os.getcwd(), 'minlplib')

    filename = os.path.join(instancedata_download_dir, 'instancedata.csv')

    if not os.path.exists(instancedata_download_dir):
        os.makedirs(instancedata_download_dir)

    if not os.path.exists(filename):
        get_minlplib_instancedata(download_dir=instancedata_download_dir)

    if acceptable_formats is None:
        acceptable_formats = set(['ams', 'gms', 'lp', 'mod', 'nl', 'osil', 'pip'])
    elif isinstance(acceptable_formats, str):
        acceptable_formats = set([acceptable_formats])
    elif isinstance(acceptable_formats, Iterable):
        acceptable_formats = set(acceptable_formats)
    else:
        raise ValueError('unrecognized type for acceptable_formats: {0}'.format(acceptable_formats))
        

    if acceptable_probtype is None:
        acceptable_probtype = set()
        for pre in ['B', 'I', 'MI', 'MB', 'S', '']:
            for post in ['NLP', 'QCQP', 'QP', 'QCP', 'P']:
                acceptable_probtype.add(pre + post)
    elif isinstance(acceptable_probtype, str):
        acceptable_probtype = set([acceptable_probtype])
    elif isinstance(acceptable_probtype, Iterable):
        acceptable_probtype = set(acceptable_probtype)
    else:
        raise ValueError('unrecognized type for acceptable_probtype: {0}'.format(acceptable_probtype))

    if acceptable_objtype is None:
        acceptable_objtype = set(['constant', 'linear', 'quadratic', 'polynomial', 'signomial', 'nonlinear'])
    elif isinstance(acceptable_objtype, str):
        acceptable_objtype = set([acceptable_objtype])
    elif isinstance(acceptable_objtype, Iterable):
        acceptable_objtype = set(acceptable_objtype)
    else:
        raise ValueError('unrecognized type for acceptable_objtype: {0}'.format(acceptable_objtype))

    if acceptable_objcurvature is None:
        acceptable_objcurvature = set(['linear', 'convex', 'concave', 'indefinite', 'nonconvex', 'nonconcave', 'unknown'])
    elif isinstance(acceptable_objcurvature, str):
        acceptable_objcurvature = set([acceptable_objcurvature])
    elif isinstance(acceptable_objcurvature, Iterable):
        acceptable_objcurvature = set(acceptable_objcurvature)
    else:
        raise ValueError('unrecognized type for acceptable_objcurvature: {0}'.format(acceptable_objcurvature))

    if acceptable_conscurvature is None:
        acceptable_conscurvature = set(['linear', 'convex', 'concave', 'indefinite', 'nonconvex', 'nonconcave', 'unknown'])
    elif isinstance(acceptable_conscurvature, str):
        acceptable_conscurvature = set([acceptable_conscurvature])
    elif isinstance(acceptable_conscurvature, Iterable):
        acceptable_conscurvature = set(acceptable_conscurvature)
    else:
        raise ValueError('unrecognized type for acceptable_conscurvature: {0}'.format(acceptable_conscurvature))

    if acceptable_convex is None:
        acceptable_convex = set(['True', 'False', ''])
    elif isinstance(acceptable_convex, str):
        acceptable_convex = set([acceptable_convex])
    elif isinstance(acceptable_convex, Iterable):
        acceptable_convex = set(str(i) for i in acceptable_convex)
    else:
        acceptable_convex = set([str(acceptable_convex)])

    csv_file = open(filename, 'r')
    reader = csv.reader(csv_file, delimiter=';')
    headings = {column: ndx for ndx, column in enumerate(next(reader))}
    rows = [row for row in reader]
    csv_file.close()

    cases = list()
    for ndx, row in enumerate(rows):
        if len(row) == 0:
            continue
        
        case_name = row[headings['name']]
        
        available_formats = row[headings['formats']]
        available_formats = available_formats.replace('set([', '')
        available_formats = available_formats.replace('])', '')
        available_formats = available_formats.replace(' ', '')
        available_formats = available_formats.replace("'", '')
        available_formats = available_formats.split(',')
        available_formats = set(available_formats)

        if len(acceptable_formats.intersection(available_formats)) == 0:
            logger.debug('excluding {case} due to available_formats'.format(case=case_name))
            continue

        probtype = row[headings['probtype']]
        if probtype not in acceptable_probtype:
            logger.debug('excluding {case} due to acceptable_probtype'.format(case=case_name))
            continue

        objtype = row[headings['objtype']]
        if objtype not in acceptable_objtype:
            logger.debug('excluding {case} due to acceptable_objtype'.format(case=case_name))
            continue

        objcurvature = row[headings['objcurvature']]
        if objcurvature not in acceptable_objcurvature:
            logger.debug('excluding {case} due to acceptable_objcurvature'.format(case=case_name))
            continue

        conscurvature = row[headings['conscurvature']]
        if conscurvature not in acceptable_conscurvature:
            logger.debug('excluding {case} due to acceptable_conscurvature'.format(case=case_name))
            continue

        convex = row[headings['convex']]
        if convex not in acceptable_convex:
            logger.debug('excluding {case} due to acceptable_convex'.format(case=case_name))
            continue

        nvars = int(row[headings['nvars']])
        if nvars < min_nvars or nvars > max_nvars:
            logger.debug('excluding {case} due to nvars'.format(case=case_name))
            continue

        nbinvars = int(row[headings['nbinvars']])
        if nbinvars < min_nbinvars or nbinvars > max_nbinvars:
            logger.debug('excluding {case} due to nbinvars'.format(case=case_name))
            continue

        nintvars = int(row[headings['nintvars']])
        if nintvars < min_nintvars or nintvars > max_nintvars:
            logger.debug('excluding {case} due to nintvars'.format(case=case_name))
            continue

        nnlvars = int(row[headings['nnlvars']])
        if nnlvars < min_nnlvars or nnlvars > max_nnlvars:
            logger.debug('excluding {case} due to nnlvars'.format(case=case_name))
            continue

        nnlbinvars = int(row[headings['nnlbinvars']])
        if nnlbinvars < min_nnlbinvars or nnlbinvars > max_nnlbinvars:
            logger.debug('excluding {case} due to nnlbinvars'.format(case=case_name))
            continue

        nnlintvars = int(row[headings['nnlintvars']])
        if nnlintvars < min_nnlintvars or nnlintvars > max_nnlintvars:
            logger.debug('excluding {case} due to nnlintvars'.format(case=case_name))
            continue

        nobjnz = int(row[headings['nobjnz']])
        if nobjnz < min_nobjnz or nobjnz > max_nobjnz:
            logger.debug('excluding {case} due to nobjnz'.format(case=case_name))
            continue

        nobjnlnz = int(row[headings['nobjnlnz']])
        if nobjnlnz < min_nobjnlnz or nobjnlnz > max_nobjnlnz:
            logger.debug('excluding {case} due to nobjnlnz'.format(case=case_name))
            continue

        ncons = int(row[headings['ncons']])
        if ncons < min_ncons or ncons > max_ncons:
            logger.debug('excluding {case} due to ncons'.format(case=case_name))
            continue

        nlincons = int(row[headings['nlincons']])
        if nlincons < min_nlincons or nlincons > max_nlincons:
            logger.debug('excluding {case} due to nlincons'.format(case=case_name))
            continue

        nquadcons = int(row[headings['nquadcons']])
        if nquadcons < min_nquadcons or nquadcons > max_nquadcons:
            logger.debug('excluding {case} due to nquadcons'.format(case=case_name))
            continue

        npolynomcons = int(row[headings['npolynomcons']])
        if npolynomcons < min_npolynomcons or npolynomcons > max_npolynomcons:
            logger.debug('excluding {case} due to npolynomcons'.format(case=case_name))
            continue

        nsignomcons = int(row[headings['nsignomcons']])
        if nsignomcons < min_nsignomcons or nsignomcons > max_nsignomcons:
            logger.debug('excluding {case} due to nsignomcons'.format(case=case_name))
            continue

        ngennlcons = int(row[headings['ngennlcons']])
        if ngennlcons < min_ngennlcons or ngennlcons > max_ngennlcons:
            logger.debug('excluding {case} due to ngennlcons'.format(case=case_name))
            continue

        njacobiannz = int(row[headings['njacobiannz']])
        if njacobiannz < min_njacobiannz or njacobiannz > max_njacobiannz:
            logger.debug('excluding {case} due to njacobiannz'.format(case=case_name))
            continue

        njacobiannlnz = int(row[headings['njacobiannlnz']])
        if njacobiannlnz < min_njacobiannlnz or njacobiannlnz > max_njacobiannlnz:
            logger.debug('excluding {case} due to njacobiannlnz'.format(case=case_name))
            continue

        nlaghessiannz = int(row[headings['nlaghessiannz']])
        if nlaghessiannz < min_nlaghessiannz or nlaghessiannz > max_nlaghessiannz:
            logger.debug('excluding {case} due to nlaghessiannz'.format(case=case_name))
            continue

        nlaghessiandiagnz = int(row[headings['nlaghessiandiagnz']])
        if nlaghessiandiagnz < min_nlaghessiandiagnz or nlaghessiandiagnz > max_nlaghessiandiagnz:
            logger.debug('excluding {case} due to nlaghessiandiagnz'.format(case=case_name))
            continue

        nsemi = int(row[headings['nsemi']])
        if nsemi < min_nsemi or nsemi > max_nsemi:
            logger.debug('excluding {case} due to nsemi'.format(case=case_name))
            continue

        nnlsemi = int(row[headings['nnlsemi']])
        if nnlsemi < min_nnlsemi or nnlsemi > max_nnlsemi:
            logger.debug('excluding {case} due to nnlsemi'.format(case=case_name))
            continue

        nsos1 = int(row[headings['nsos1']])
        if nsos1 < min_nsos1 or nsos1 > max_nsos1:
            logger.debug('excluding {case} due to nsos1'.format(case=case_name))
            continue

        nsos2 = int(row[headings['nsos2']])
        if nsos2 < min_nsos2 or nsos2 > max_nsos2:
            logger.debug('excluding {case} due to nsos2'.format(case=case_name))
            continue

        cases.append(case_name)

    return cases


def get_minlplib(download_dir=None, format='osil'):
    """
    Download MINLPLib

    Parameters
    ----------
    download_dir: str
        The directory in which to place the downloaded files. The default will be a 
        current_working_directory/minlplib/file_format/.
    format: str
        The file format requested. Options are ams, gms, lp, mod, nl, osil, and pip
    """
    if download_dir is None:
        download_dir = os.path.join(os.getcwd(), 'minlplib', format)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    downloader = download.FileDownloader()
    zip_filename = os.path.join(download_dir, 'minlplib_'+format+'.zip')
    downloader.set_destination_filename(zip_filename)
    downloader.get_binary_file('http://www.minlplib.org/minlplib_'+format+'.zip')
    zipper = ZipFile(zip_filename, 'r')
    zipper.extractall(download_dir)
    os.remove(zip_filename)
    for i in os.listdir(os.path.join(download_dir, 'minlplib', format)):
        os.rename(os.path.join(download_dir, 'minlplib', format, i), os.path.join(download_dir, i))
    os.rmdir(os.path.join(download_dir, 'minlplib', format))
    os.rmdir(os.path.join(download_dir, 'minlplib'))
