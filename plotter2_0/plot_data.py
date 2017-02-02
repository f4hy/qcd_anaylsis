import logging


import inspect
import sys
import plot_light_masses        # noqa
import plot_latparams           # noqa
import plot_heavy_decays        # noqa
import plot_heavy_masses        # noqa
import plot_fdsqrtmd            # noqa
import plot_fdssqrtmds          # noqa
import plot_interpolated_fsqrtm # noqa

# function_list = inspect.getmembers(sys.modules["fitfunctions"], inspect.isclass)
# functions = {name: f for name, f in function_list}


class plot_data(object):

    def __init__(self, value, error, label=None, physical=None):
        self.value = value
        self.error = error
        self.label = label
        if label is None:
            self.label = ""
        self.physical = physical
        if physical is None:
            self.physical = {}


def package_heavies(datas):
    pass


def get_data(ed, data_type, options):

    # This is hackish but makes the whole thing work
    # We create a map of function names to their functions themselves
    # Mostly to break up these functions into many files
    modules_with_plot_fucntions = ("plot_latparams",  # List of modules with the plot files
                                   "plot_light_masses",
                                   "plot_heavy_decays", "plot_heavy_masses",
                                   "plot_fdsqrtmd", "plot_fdssqrtmds", "plot_interpolated_fsqrtm")

    function_map = {}
    for m in modules_with_plot_fucntions:
        function_map.update(dict(inspect.getmembers(sys.modules["plotter2_0." + m], inspect.isfunction)))

    def dataindex():
        num = 0
        while num < 100:
            yield num
            num += 1

    if data_type in function_map:
        result = function_map[data_type](ed, options)
        # result will either be tuple and we make it a plot_data
        # or a dictionary where we make each element a plot_data
        if isinstance(result, tuple):
            return plot_data(*result)
        else:
            return {k: plot_data(*v) for k,v in result.iteritems()}
    else:
        logging.error("{} not supported as a data type yet".format(data_type))
        logging.error("supported_types: {}".format(sorted(function_map)))
        raise RuntimeError("{} not supported as a data type yet".format(data_type))




    raise RuntimeError("{} not supported as a data type yet".format(data_type))
