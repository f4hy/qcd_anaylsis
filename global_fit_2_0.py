#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
from iminuit import Minuit

import inspect
import os

from misc import all_equal
from commonplotlib.progress_bar import progress_bar
from ensemble_data2_0.all_ensemble_data import ensemble_data
from global_model2_0.global_fit_model2_0 import Model

from global_model2_0.fdssqrtms_models import * # noqa
from global_model2_0.single_heavy_fd_models import * # noqa
from global_model2_0.fd_models import * # noqa
from global_model2_0.pion_chiral_model import * # noqa
from global_model2_0.kaon_chiral_model import * # noqa


def interpolate(data, model_str, options):

    logging.info("Fitting data")

    valid_models = {m.__name__: m for m in Model.__subclasses__()}
    logging.debug("valid models available {}".format(valid_models.keys()))
    model_obj = valid_models[model_str](data, options, scale_sigma=options.shift_scale)

    params = model_obj.params
    model_fun = model_obj.sqr_diff

    # params, model_fun = model_obj.build_function()
    ARGS = inspect.getargspec(model_fun).args[1:]
    logging.info("Params {}".format(params))
    bootstraps = [d.shape[1] for d in model_obj.data.values()]
    assert(all_equal(bootstraps))
    N = bootstraps[0]

    dof = model_obj.degrees_of_freedom()

    if dof < 1.0:
        raise RuntimeError("dof < 1")

    logging.info("fitting mean")
    model_obj.boostrap = "mean"
    mean_m = Minuit(model_fun, errordef=1.0, print_level=0, pedantic=True, **params)
    mean_m.set_strategy(2)
    mean_results = mean_m.migrad()
    logging.debug(mean_results)

    logging.info("chi^2={}, dof={}, chi^2/dof={}".format(mean_m.fval, dof, mean_m.fval / dof))
    logging.debug('covariance {}'.format(mean_m.covariance))
    logging.info('fitted values {}'.format(mean_m.values))
    logging.info('fitted errors {}'.format(mean_m.errors))

    if not mean_m.get_fmin().is_valid:
        logging.error("NOT VALID")
        exit(-1)

    params.update(mean_m.values)
    if options.scale_systematic:
        logging.info("Finding the systematic dependance upon the scale seeting")
        find_scale_dependance(data, valid_models[model_str], mean_m.values, options)
        exit(0)
    # return mean_m, {0: mean_m}, np.nan
    if (mean_m.fval / dof) > 500.0:
        logging.error("Chi^2/dof is huge, dont bother with bootstraps")
        return mean_m, {0: mean_m}, np.nan

    bootstrap_m = {}
    logging.info("Fitting {} bootstraps".format(N))
    progressb = progress_bar(N)
    for b in range(N):
        progressb.update(b)
        model_obj.set_bootstrap(b)
        bootstrap_m[b] = Minuit(model_fun, errordef=1.0, print_level=0, pedantic=True, **params)
        bootstrap_m[b].set_strategy(1)
        bootstrap_results = bootstrap_m[b].migrad()
        logging.debug(bootstrap_results)
        if not bootstrap_m[b].get_fmin().is_valid:
            logging.error("NOT VALID for bootstrap {}".format(b))
            exit(-1)
    progressb.done()

    logging.info('fitted mean values {}'.format(mean_m.values))
    logging.info('fitted mean errors {}'.format(mean_m.errors))
    means = []
    for i in ARGS:
        x = [b.values[i] for b in bootstrap_m.values()]
        ex = [b.errors[i] for b in bootstrap_m.values()]
        means.append(np.mean(x))
        logging.info("bootstraped {}: mean {} med {} std {}".format(i, np.mean(x), np.median(x), np.std(x)))
        logging.info("bootstraped error {}: mean {} med {} std {}".format(i, np.mean(ex), np.median(ex), np.std(ex)))

        logging.info("bootstraped chi^2 {}: dof {} chi/dof std {}".format(i, np.mean(ex), np.median(ex), np.std(ex)))

    fvals = [b.fval/dof for b in bootstrap_m.values()]

    boot_ave_fval = model_fun(*means)
    return mean_m, bootstrap_m, np.mean(fvals), dof


def find_scale_dependance(data, model, mean, options):
    model_obj_p = model(data, options, scale_sigma=+1)
    model_obj_m = model(data, options, scale_sigma=-1)

    params_p = model_obj_p.params
    params_m = model_obj_m.params
    model_fun_p = model_obj_p.sqr_diff
    model_fun_m = model_obj_m.sqr_diff

    logging.info("fitting mean_p")
    model_obj_p.boostrap = "mean"
    mean_m_p = Minuit(model_fun_p, errordef=1.0, print_level=0, pedantic=True, **params_p)
    mean_m_p.set_strategy(2)
    mean_results_p = mean_m_p.migrad()
    logging.debug(mean_results_p)

    logging.info("a+sigma chi^2={}".format(mean_m_p.fval))
    logging.info('a+sigma fitted values {}'.format(mean_m_p.values))
    logging.info('a+sigma fitted errors {}'.format(mean_m_p.errors))


    logging.info("fitting mean_m")
    model_obj_m.boostrap = "mean"
    mean_m_m = Minuit(model_fun_m, errordef=1.0, print_level=0, pedantic=True, **params_m)
    mean_m_m.set_strategy(2)
    mean_results_m = mean_m_m.migrad()
    logging.debug(mean_results_m)

    logging.info("a+sigma chi^2={}".format(mean_m_m.fval))
    logging.info('a+sigma fitted values {}'.format(mean_m_m.values))
    logging.info('a+sigma fitted errors {}'.format(mean_m_m.errors))

    logging.info("a+sigma difference = {}".format({k: mean[k] - mean_m_p.values[k]
                                                  for k in mean.keys()}))
    logging.info("a-sigma difference = {}".format({k: mean[k] - mean_m_m.values[k]
                                                  for k in mean.keys()}))
    maxdiff = {k: max(abs(mean[k] - mean_m_m.values[k]),abs(mean[k] - mean_m_p.values[k])) for k in mean.keys()}
    logging.info("a1sigma difference = {}".format(maxdiff))
    return maxdiff


def write_data(fit_parameters, output_stub, suffix, model, dof):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    ensure_dir(outfilename)
    logging.info("writing fit to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        chisqrbydof = fit_parameters.fval / dof
        ofile.write("#{} chisqr {}, dof {}, chisqr/dof {}\n".format(model, fit_parameters.fval,
                                                                    dof,
                                                                    chisqrbydof))

        for name in fit_parameters.values:
            ofile.write("{}, {} +/- {}\n".format(name, fit_parameters.values[name],
                                                 fit_parameters.errors[name]))


def write_bootstrap_data(fit_parameters, boot_fval, output_stub, suffix, model, dof):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    ensure_dir(outfilename)
    logging.info("writing bootstrapfit to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        fval = boot_fval
        chisqrbydof = fval / dof
        ofile.write("#{} chisqr {}, dof {}, chisqr/dof {}\n".format(model, fval, dof, chisqrbydof))

        for name in fit_parameters[0].values:
            values = [b.values[name] for b in fit_parameters.values()]
            value = np.mean(values)
            error = np.std(values)
            ofile.write("{}, {} +/- {}\n".format(name, value, error))
    bootstraps_filename = output_stub + ".boot"
    logging.info("writing bootstraps of fit to {}".format(bootstraps_filename))
    with open(bootstraps_filename, "w") as ofile:
        names = fit_parameters[0].values.keys()
        ofile.write("# " + model + "," + ",".join(names) + "\n")
        for b, d in fit_parameters.iteritems():
            line = ",".join(["{}".format(d.values[n]) for n in names])
            ofile.write("{}\n".format(line))

def ensure_dir(filename):
    outdir = os.path.dirname(filename)
    if not os.path.exists(outdir):
        logging.info("directory for output {} does not exist, atempting to create".format(outdir))
        if outdir is not "":
            os.makedirs(outdir)



def global_fit(options):
    """ perform a global fit """
    logging.debug("Called with {}".format(options))

    ensembles = []
    for es in options.ensembles:
        try:
            ed = ensemble_data(es)
            if ed.ep.ename != "KC0":
                ensembles.append(ed)
        except Exception as e:
            logging.error("exception in loading ensemble_data")
            raise e
            raise argparse.ArgumentTypeError("Argument {} does not have valid ensemble data".format(es))

    mean_fit_parameters, bootstrap_fit_parameters, boot_fval, dof = interpolate(ensembles, options.model, options)
    write_data(mean_fit_parameters, options.output_stub, ".fit", options.model, dof)
    write_bootstrap_data(bootstrap_fit_parameters, boot_fval, options.output_stub, "bootstraped.fit", options.model, dof)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="script to interpolate the heavy mass")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("--pdf", action="store_true",
                        help="produce a pdf instead of a png")
    parser.add_argument("--seperate_strange", action="store_true",
                        help="fit different strange values seperately")
    parser.add_argument("-o", "--output_stub", type=str, required=False,
                        help="stub of name to write output to")
    parser.add_argument('ensembles', metavar='f', type=str, nargs='+',
                        help='files to plot')
    parser.add_argument("--fitdata", required=False, type=str,
                        help="folder for fitdata when needed")
    parser.add_argument("--mpi_cutoff", required=False, type=float,
                        help="cutoff value for mpi")
    parser.add_argument("--hqm_cutoff", required=False, type=float, default=100000.0,
                        help="cutoff value for heavy quark mass")
    parser.add_argument("-m", "--model", required=False, type=str, default="linear_FD_in_mpi",
                        help="which model to use")
    parser.add_argument("--scale_systematic", required=False, action="store_true",
                        help="find the systematic from changing the lat spacing by 1 sigma")
    parser.add_argument("--shift_scale", required=False, type=int, choices=[-1, 0, 1], default=0,
                        help="Shift scale by 1 or -1 sigma before fitting")
    parser.add_argument("-z", "--zero", required=False, type=str, nargs='+', default=[],
                        help="Zero a fit variable")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    ensure_dir(args.output_stub)

    if args.output_stub is not None:
        root = logging.getLogger()
        errfilename = args.output_stub+".err"
        errfilehandler = logging.FileHandler(errfilename, delay=True)
        errfilehandler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        errfilehandler.setFormatter(formatter)
        root.addHandler(errfilehandler)
        logfilename = args.output_stub+".log"
        logfilehandler = logging.FileHandler(logfilename, delay=True)
        logfilehandler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        logfilehandler.setFormatter(formatter)
        root.addHandler(logfilehandler)


    global_fit(args)
