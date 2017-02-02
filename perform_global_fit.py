#!/usr/bin/env python2
import logging                  # Including many defaults, can be removed if unneeded
import argparse
import numpy as np
from iminuit import Minuit

import inspect

# from global_fit_model import Model

from misc import all_equal
from commonplotlib.progress_bar import progress_bar
from all_ensemble_data import ensemble_data
import new_fit_model


def interpolate(data, model_str, options):

    logging.info("Fitting data")

    model_obj = getattr(new_fit_model, model_str)(data, options)
    # exit(-1)
    # model_obj = Model(data, model_str, options)
    params = model_obj.params
    model_fun = model_obj.sqr_diff

    # params, model_fun = model_obj.build_function()
    ARGS = inspect.getargspec(model_fun).args[1:]
    logging.info("Params {}".format(params))

    bootstraps = [d.shape[1] for d in model_obj.data.values()]
    assert(all_equal(bootstraps))
    N = bootstraps[0]
    datapoints = [d.shape[0] for d in model_obj.data.values()]
    assert(all_equal(datapoints))
    ndata = datapoints[0]

    fixed_parms = [p for p in params if "fix" in p and params[p]]
    Nfree_params = len(ARGS) - len(fixed_parms)
    if model_str.startswith("combined"):
        dof = float(ndata * 2 - Nfree_params)
    else:
        dof = float(ndata - Nfree_params)

    logging.info("DOF {}, data {}, free {}".format(dof, len(data), Nfree_params))

    if dof < 1.0:
        raise RuntimeError("dof < 1")

    logging.info("fitting mean")
    model_obj.boostrap = "mean"
    mean_m = Minuit(model_fun, errordef=dof, print_level=0, pedantic=True, **params)
    mean_m.set_strategy(2)
    mean_results = mean_m.migrad()
    logging.debug(mean_results)

    logging.info("chi^2={}, dof={}, chi^2/dof={}".format(mean_m.fval, dof, mean_m.fval / dof))
    logging.info('covariance {}'.format(mean_m.covariance))
    logging.info('fitted values {}'.format(mean_m.values))
    logging.info('fitted errors {}'.format(mean_m.errors))

    if not mean_m.get_fmin().is_valid:
        logging.error("NOT VALID")
        exit(-1)

    params.update(mean_m.values)

    # return mean_m, {0: mean_m}, np.nan
    if (mean_m.fval / dof) > 100.0:
        logging.error("Chi^2/dof is huge, dont bother with bootstraps")
        return mean_m, {0: mean_m}, np.nan

    bootstrap_m = {}
    progressb = progress_bar(N)
    for b in range(N):
        progressb.update(b)
        model_obj.set_bootstrap(b)
        bootstrap_m[b] = Minuit(model_fun, errordef=dof, print_level=0, pedantic=True, **params)
        bootstrap_m[b].set_strategy(2)
        bootstrap_results = bootstrap_m[b].migrad()
        logging.debug(bootstrap_results)
        if not bootstrap_m[b].get_fmin().is_valid:
            logging.error("NOT VALID for bootstrap".format(b))
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

    boot_ave_fval = model_fun(*means)
    return mean_m, bootstrap_m, boot_ave_fval


def write_data(fit_parameters, output_stub, suffix, model):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing fit to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        chisqrbydof = fit_parameters.fval / fit_parameters.errordef
        ofile.write("#{} chisqr {}, dof {}, chisqr/dof {}\n".format(model, fit_parameters.fval,
                                                                    fit_parameters.errordef,
                                                                    chisqrbydof))

        for name in fit_parameters.values:
            ofile.write("{}, {} +/- {}\n".format(name, fit_parameters.values[name],
                                                 fit_parameters.errors[name]))


def write_bootstrap_data(fit_parameters, boot_fval, output_stub, suffix, model):
    if output_stub is None:
        logging.info("Not writing output")
        return
    outfilename = output_stub + suffix
    logging.info("writing bootstrapfit to {}".format(outfilename))
    with open(outfilename, "w") as ofile:
        fval = boot_fval
        dof = np.mean([b.errordef for b in fit_parameters.values()])
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


def global_fit(options):
    """ perform a global fit """
    logging.debug("Called with {}".format(options))

    ensembles = []
    for es in options.ensembles:
        try:
            ed = ensemble_data(es)
            ensembles.append(ed)
        except Exception as e:
            logging.error("exception in loading ensemble_data")
            raise e
            raise argparse.ArgumentTypeError("Argument {} does not have valid ensemble data".format(es))

    mean_fit_parameters, bootstrap_fit_parameters, boot_fval = interpolate(ensembles, options.model, options)

    write_data(mean_fit_parameters, options.output_stub, ".fit", options.model)
    write_bootstrap_data(bootstrap_fit_parameters, boot_fval, options.output_stub, "bootstraped.fit", options.model)


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
    parser.add_argument("--cutoff", required=False, type=float,
                        help="cutoff value")
    parser.add_argument("--hqm_cutoff", required=False, type=float, default=100000.0,
                        help="cutoff value")
    parser.add_argument("-m", "--model", required=False, type=str, default="linear_FD_in_mpi",
                        help="which model to use")
    parser.add_argument("-z", "--zero", required=False, type=str, nargs='+', default=[],
                        help="Zero a fit variable")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    global_fit(args)
