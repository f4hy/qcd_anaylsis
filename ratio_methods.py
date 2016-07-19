#!/usr/bin/env python2
import logging
import numpy as np

Lambda = 1.25
Dmass = 1866.0
Dmass = 1896.0
mc = 1080.0
fd = 212.0


def ratioy(mu, z, z2):
    return 1 + z / mu + z2 / (mu**2)


def ratio_chain(fittype, values):

    if fittype in ["mD_renorm_ratio", "m_mq_ma_ratio"]:
        md_chain(values)

    if fittype in ["fdsqrtm_mq_ma_ratio"]:
        fd_chain(values)


    logging.info("Chain not implimted for {}".format(fittype))


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
    exit()

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

        fdfile.write("{}, {}\n".format(m_qip, product * product * fd / np.sqrt(Lambda**k)))
        fdsqrtmfile.write("{}, {}\n".format(m_qip, product * fd * np.sqrt(Dmass)))

    logging.info("Writing chain computed values to {}".format(outfilename))
    fdfile.close()
    fdsqrtmfile.close()
    # fig2.show()
    exit()
