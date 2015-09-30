#!/usr/bin/env python2

# def residual_mass(udmass, smass):
#     residual_masses = {(0.019, 0.030): 0.00047, (0.012, 0.030): 0.00040, (0.007, 0.030): 0.00042,
#                        (0.019, 0.040): 0.00041, (0.012, 0.040): 0.00044, (0.007, 0.040): 0.00038,
#                        (0.0035, 0.040): 0.00039,
#                        (0.008, 0.018): 0.00000, (0.008, 0.025): 0.00000,
#                        (0.012, 0.018): 0.000053, (0.012, 0.025): 0.000032,
#                        (0.0042, 0.018): 0.00000, (0.0042, 0.025): 0.00000 ,
#                        (0.0030, 0.0150): 0.00000}

#     try:
#         return residual_masses[(udmass, smass)]
#     except KeyError:
#         return 0.0



def residual_mass(dp):
    residual_masses = {"4.17":  0.00041, "4.35": 0.00005, "4.47": 0.0 }

    return residual_masses[dp.beta]

def residual_mass_errors(dp):
    residual_mass_errors = {"4.17":  0.00008, "4.35": 0.00003, "4.47": 0.0 }

    return residual_mass_errors[dp.beta]
