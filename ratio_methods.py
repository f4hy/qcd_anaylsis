#!/usr/bin/env python2
import logging
import argparse
import numpy as np


from msbar_convert import get_matm

from alpha_s import get_Cmu_mbar

Lambda = 1.25
Dmass = 1866.0
Dmass = 1896.0
mc = 1080.0
fd = 212.0

matching_factor=  0.807

def ratioy(mu, z, z2):
    return 1 + z / mu + z2 / (mu**2)


def ratio_chain(fittype, values):

    if fittype in ["mD_renorm_ratio", "m_mq_ma_ratio"]:
        return md_chain(values)

    if fittype in ["mD_corrected_pole_ratio"]:
        return md_pole_chain(values)


    if fittype in ["fdsqrtm_mq_ma_ratio"]:
        return fd_chain(values)

    if fittype in ["fdsqrtmd_matched_ratio"]:
        return fd_matched_chain(values)


    logging.info("Chain not implimted for {}".format(fittype))
    exit(-1)

def md_chain(values):

    z = values["z"]
    z2 = values["z2"]

    product = 1.0

    outfilename = "md_chain.txt"

    # fig2 = plt.figure()
    # maxe = fig2.add_subplot(111)
    mdfile = open(outfilename, 'w')
    for k in range(0, 8):
        m_qi = (Lambda**k) * mc
        m_qip = (Lambda**(k+1)) * mc
        r_i = ratioy(m_qi, z, z2)
        logging.info("{} {} {}".format(k, 1.25**k, mc * (1.25**k)))
        logging.info("ratio {}".format(r_i))
        product *= r_i
        logging.info("m_qi, product, {}, {}".format(m_qi, product))
        # maxe.scatter((Lambda**k) * mc, product * (Lambda**k) * Dmass)

        mdfile.write("{}, {}\n".format(m_qip, product * (Lambda**(k+1)) * Dmass))

    logging.info("Writing chain computed values to {}".format(outfilename))
    mdfile.close()
    # fig2.show()
    #exit()

def md_pole_chain(values):

    z = values["z"]
    z2 = values["z2"]

    product = 1.0

    outfilename = "md_pole_chain.txt"

    # fig2 = plt.figure()
    # maxe = fig2.add_subplot(111)
    mdfile = open(outfilename, 'w')
    for k in range(0, 8):
        m_qi = (Lambda**k) * mc
        m_qip = (Lambda**(k+1)) * mc
        r_i = ratioy(m_qi, z, z2)
        logging.info("{} {} {}".format(k, 1.25**k, mc * (1.25**k)))
        logging.info("ratio {}".format(r_i))
        product *= r_i
        logging.info("m_qi, product, {}, {}".format(m_qi, product))
        # maxe.scatter((Lambda**k) * mc, product * (Lambda**k) * Dmass)
        print m_qi, get_matm(m_qi, m_qi)
        print m_qip, get_matm(m_qip, m_qip)
        mdfile.write("{}, {}\n".format(m_qip, product *  Dmass * (get_matm(m_qip, m_qip) / get_matm(mc, mc) ) ))

    logging.info("Writing chain computed values to {}".format(outfilename))
    mdfile.close()
    # fig2.show()
    #exit()


def fd_chain(values):

    z = values["z"]
    z2 = values["z2"]

    product = 1.0

    outfilename = "fhl_chain.txt"
    fmoutfilename = "fhl_sqrtm_chain.txt"

    # fig2 = plt.figure()
    # maxe = fig2.add_subplot(111)
    fdfile = open(outfilename, 'w')
    fdsqrtmfile = open(fmoutfilename, 'w')
    for k in range(0, 8):
        m_qi = (Lambda**k) * mc
        m_qip = (Lambda**(k+1)) * mc
        r_i = ratioy(m_qi, z, z2)
        logging.info("{} {} {}".format(k, 1.25**k, mc * (1.25**k)))
        logging.info("ratio {}".format(r_i))
        product *= r_i
        logging.info("m_qi, product, {}, {}".format(m_qi, product))
        logging.info("m_qi, f, {}, {}".format(m_qi, product * fd / np.sqrt(Lambda**k)))
        logging.info("m_qi, fsqrtm, {}, {}".format(m_qi, product * fd * np.sqrt(Dmass)))
        logging.info("m_qi, fsqrtm / mb, {}, {}".format(m_qi, product * fd * np.sqrt(Dmass) / np.sqrt(5174.0)))
        # maxe.scatter((Lambda**k) * mc, product * (Lambda**k) * Dmass)

        # fdfile.write("{}, {}\n".format(m_qip,  product * fd / np.sqrt(Lambda**k)))
        fdsqrtmfile.write("{}, {}\n".format(m_qip, product * (fd * np.sqrt(Dmass)) ))

    logging.info("Writing chain computed values to {}".format(outfilename))
    fdfile.close()
    fdsqrtmfile.close()
    # fig2.show()
    #exit()

def fd_matched_chain(values):

    z = values["z"]
    z2 = values["z2"]

    product = 1.0

    fmoutfilename = "fhl_sqrtm_matched_chain.txt"

    # fig2 = plt.figure()
    # maxe = fig2.add_subplot(111)
    fdsqrtmfile = open(fmoutfilename, 'w')
    for k in range(0, 8):
        m_qi = (Lambda**k) * mc
        m_qip = (Lambda**(k+1)) * mc
        r_i = ratioy(m_qi, z, z2)
        product *= r_i

        fdsqrtmfile.write("{}, {}\n".format(m_qip, product * (fd * np.sqrt(Dmass))/matching_factor ))

    logging.info("Writing chain computed values to {}".format(fmoutfilename))
    fdsqrtmfile.close()
    # fig2.show()
    #exit()

def main(options):

    values = {"z": -32.16, "z2": 21502.1827382 }
    fd_matched_chain(values)
    md_pole_chain(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chain equations")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
        logging.debug("Verbose debuging mode activated")
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    main(args)
