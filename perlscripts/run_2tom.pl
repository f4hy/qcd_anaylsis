#!/usr/bin/perl -w
use POSIX;

$m0 = $ARGV[0]; # initial mass m(2GeV) in GeV
$m1 = $ARGV[1]; # target mass m(m)

$mu = 2.0; # in GeV

$Nf = 3;
$Lambda = 0.340; # in GeV

## coupling constant
$pi = 3.1415927;
$zeta3 = 1.2020569032; #$zeta5 = 1.0369277551;
$beta0 = 1/(4) * (11 - 2/3*$Nf);
$beta1 = 1/(4)**2 * (102 - 38/3*$Nf);
$beta2 = 1/(4)**3 * (2857/2 - 5033/18*$Nf + 325/54*$Nf**2);
$beta3 = 1/(4)**4 * ( 149753/6 + 3564*$zeta3 -
                   (1078361/162 + 6508/27*$zeta3)*$Nf +
                   (50065/162 + 6472/81*$zeta3)*$Nf**2 + 1093/729*$Nf**3 );
#print "# b0, b1, b2, b3 = $beta0 $beta1 $beta2 $beta3\n";

# alpha_s(mu)
$L = log($mu**2/$Lambda**2);
$alpha1 = $pi/($beta0*$L);
$alpha2 = $pi/($beta0*$L) * (-$beta1/$beta0**2 * log($L)/$L);
$alpha3 = $pi/($beta0*$L) * 1/($beta0**2*$L**2)*(
    $beta1**2/$beta0**2 * ((log($L))**2 - log($L) - 1) + $beta2/$beta0);
$alpha4 = $pi/($beta0*$L) * 1/($beta0**3*$L**3)*(
    $beta1**3/$beta0**3 * (-(log($L))**3 + 5/2*(log($L))**2 + 2*log($L) - 1/2)
    -3*$beta1*$beta2/$beta0**2 * log($L) + $beta3/(2*$beta0) );
$alpha = $alpha1 + $alpha2 + $alpha3 + $alpha4;
#print "# alpha_s (1, 2, 3, 4 loop at mMS) = $alpha1 $alpha2 $alpha3 $alpha4 = $alpha\n";
$as = $alpha/$pi;
print "alpha, $alpha, as, $as, c0 $c0 \n";

$c0 = $as**(4/9)*(1+0.895062*$as+1.37143*$as**2+1.95168*$as**3);

print "as, $as, c0 $c0 \n";

for ($iter = 0; $iter < 20; $iter++) {

# alpha_s(mu')
$L = log($m1**2/$Lambda**2);
$alpha1 = $pi/($beta0*$L);
$alpha2 = $pi/($beta0*$L) * (-$beta1/$beta0**2 * log($L)/$L);
$alpha3 = $pi/($beta0*$L) * 1/($beta0**2*$L**2)*(
    $beta1**2/$beta0**2 * ((log($L))**2 - log($L) - 1) + $beta2/$beta0);
$alpha4 = $pi/($beta0*$L) * 1/($beta0**3*$L**3)*(
    $beta1**3/$beta0**3 * (-(log($L))**3 + 5/2*(log($L))**2 + 2*log($L) - 1/2)
    -3*$beta1*$beta2/$beta0**2 * log($L) + $beta3/(2*$beta0) );
$alpha = $alpha1 + $alpha2 + $alpha3 + $alpha4;
#print "# alpha_s (1, 2, 3, 4 loop at mMS) = $alpha1 $alpha2 $alpha3 $alpha4 = $alpha\n";
$as = $alpha/$pi;

$c1 = $as**(4/9)*(1+0.895062*$as+1.37143*$as**2+1.95168*$as**3);

$m1_out = $c1/$c0 * $m0;
print "$m0 $m1 $m1_out\n";
$m1 = $m1_out;
};
