import os, sys
import logging, glob

from doit_xtended.report import template_renderer
from doit_xtended.collect import collect_functions, collect_and_inject_indices
from doit_xtended.utils import target_filenames
from doit_xtended.linkedtasks import link_targets

DOIT_CONFIG = {'default_tasks': []}

log = logging.getLogger()


@link_targets
def task_data():
    """ runs python functions in src.data creating data files """

    functions = collect_functions('src.data')

    outpath = r"data\processed"
    for func_name, func, src_file in functions:
        outfiles = target_filenames(outpath,func,suffix='_')

        yield {'name': func_name,
               'doc': func.__doc__,
               'actions': [func],
               'targets': outfiles,
               'file_dep': [src_file],
               'clean': True}

# inject data indices into doit
collect_and_inject_indices('src.data',
                           globals(),
                           indexfile=r'data/processed/{name}.pkl',
                           datapath=r'data/processd/{name}',
                           inputpath=r'data/raw')


@link_targets
def task_features():
    """ runs python functions in the src.features directory (input arguments are data/processed/arg.pkl, outfile to reports/func_name) """

    functions = collect_functions('src.features')

    outpath = r'data\processed'

    for name, func, src in functions:
        outfiles = target_filenames(outpath, func, suffix='_')

        yield {'name': name,
               'actions': [func],
               'doc': func.__doc__,
               'targets': outfiles,
               'file_dep': [src],
               'clean': True}

# inject feature indices into doit
collect_and_inject_indices('src.features',
                           globals(),
                           indexfile=r'data\processed\{name}',
                           datapath=r'data\processd\{name}',
                           inputpath=r'data\processed')

@link_targets
def task_models():
    """ preprocess data, e.g. resample the raw data in raw.h5 """

    functions = collect_functions('src.models',exclude_module='pure_tensorflow')

    outpath = r'models'

    for name, func, src in functions:
        outfiles = target_filenames(outpath, func, suffix='_')

        yield {'name': name,
               'actions': [func],
               'doc': func.__doc__,
               'targets': outfiles,
               'file_dep': [src],
               'clean': True}


@link_targets
def task_figures():
    """ runs python functions in the src.visualization directory (input arguments are data\processed\arg.pkl, outfile to reports\figures\func_name.png) """

    figures = collect_functions('src.visualization', exclude_module=['utils'])

    outpath = r'reports\figures'

    for name, func, src in figures:
        outfiles = target_filenames(outpath, func, suffix='png')

        yield {'name': name,
               'actions': [func],
               'doc': func.__doc__,
               'targets': outfiles,
               'file_dep': [src],
               'clean': True}
        
        
def task_numbers():
    """ evaluates all numbers files and join them together into reports\numbers.yaml """

    numbers_path = r'reports\numbers\*.yaml'
    context_file = r'reports\numbers.yaml'

    def join_numbers(numbers_files, outfile):
        import yaml
        context = {}
        for filepath in numbers_files:
            _, filename = os.path.split(filepath)
            name, _ = os.path.splitext(filename)
            log.info("load '%s'...",filename)
            with open(filepath,'r') as fp:
                context[name] = yaml.load(fp)

        log.info("store into '%s'...",outfile)
        with open(outfile,'w+') as fp:
            yaml.dump(context,fp)

    numbers_files = glob.glob(numbers_path)

    return {
        'actions': [(join_numbers, (numbers_files, context_file))],
        'targets': [context_file],
        'file_dep': numbers_files,
        'uptodate': [True],
        'clean': True
    }


def task_report():
    """ renders the template in reports\report_template.docx with all figures and numbers.yaml mapped"""

    figure_files = glob.glob(r'reports\figures\*.png')

    outfile = r'reports\report_ThreadImageProcessor.docx'
    infile = r'reports\report_template.docx'
    context_file = r'reports\numbers.yaml'
    src_file = 'dodo_utils.py'

    return {
        'actions': [(template_renderer(figure_files, context_file), (infile, outfile))],
        'targets': [outfile],
        'file_dep': [infile] + figure_files + [context_file, src_file],
        'clean': True
    }


from doit_xtended.linkedtasks import _generated_linked_tasks

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from doit.cmd_base import ModuleTaskLoader
    from doit.doit_cmd import DoitMain

    d = DoitMain(ModuleTaskLoader(globals()))
    d.run(['list','--all'])