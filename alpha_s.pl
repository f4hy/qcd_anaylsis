#!/usr/bin/perl -w
use POSIX;

$mu = $ARGV[0]; # in GeV

$Nf = 3;
$Lambda = 0.340; # in GeV


## coupling constant
$pi = 3.1415927;
$zeta3 = 1.2020569032; $zeta5 = 1.0369277551;
$beta0 = 1/(4) * (11 - 2/3*$Nf);
$beta1 = 1/(4)**2 * (102 - 38/3*$Nf);
$beta2 = 1/(4)**3 * (2857/2 - 5033/18*$Nf + 325/54*$Nf**2);
$beta3 = 1/(4)**4 * ( 149753/6 + 3564*$zeta3 -
                   (1078361/162 + 6508/27*$zeta3)*$Nf +
                   (50065/162 + 6472/81*$zeta3)*$Nf**2 + 1093/729*$Nf**3 );
print "# b0, b1, b2, b3 = $beta0 $beta1 $beta2 $beta3\n";

# alpha_s(mu)
$L = log($mu**2/$Lambda**2);
$alpha1 = $pi/($beta0*$L);
$alpha2 = $alpha1 + $pi/($beta0*$L) * (-$beta1/$beta0**2 * log($L)/$L);
$alpha3 = $alpha2 + $pi/($beta0*$L) * 1/($beta0**2*$L**2)*(
    $beta1**2/$beta0**2 * ((log($L))**2 - log($L) - 1) + $beta2/$beta0);
$alpha4 = $alpha3 + $pi/($beta0*$L) * 1/($beta0**3*$L**3)*(
    $beta1**3/$beta0**3 * (-(log($L))**3 + 5/2*(log($L))**2 + 2*log($L)
- 1/2)
    -3*$beta1*$beta2/$beta0**2 * log($L) + $beta3/(2*$beta0) );
print "# alpha_s (1, 2, 3, 4 loop at mu = $mu GeV) = $alpha1 $alpha2 $
alpha3 $alpha4\n";
