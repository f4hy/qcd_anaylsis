#!/usr/bin/perl -w
use POSIX;

$mu = 2.0; # in GeV
$mbar = $ARGV[0]; # MSbar mass in GeV

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

# alpha_s(mbar)
$L = log($mbar**2/$Lambda**2);
$alpha1 = $pi/($beta0*$L);
$alpha2 = $pi/($beta0*$L) * (-$beta1/$beta0**2 * log($L)/$L);
$alpha3 = $pi/($beta0*$L) * 1/($beta0**2*$L**2)*(
    $beta1**2/$beta0**2 * ((log($L))**2 - log($L) - 1) + $beta2/$beta0);
$alpha4 = $pi/($beta0*$L) * 1/($beta0**3*$L**3)*(
    $beta1**3/$beta0**3 * (-(log($L))**3 + 5/2*(log($L))**2 + 2*log($L) - 1/2)
    -3*$beta1*$beta2/$beta0**2 * log($L) + $beta3/(2*$beta0) );
$alpha = $alpha1 + $alpha2 + $alpha3 + $alpha4;
print "# alpha_s (1, 2, 3, 4 loop at mMS) = $alpha1 $alpha2 $alpha3 $alpha4 = $alpha\n";

# pole mass
$as = $alpha/$pi;
$Cm1 = 4/3; $Zm1 = $Cm1*$as;
#$Zm2 = $Zm1 + (-1.0414*$Nf+13.4434)*$as**2;
#$Zm3 = $Zm2 + (0.6527*$Nf**2 - 26.655*$Nf + 190.595)*$as**3;
$Cm2 = (6.248*$beta0-3.739); $Zm2 = $Cm2*$as**2;
$Cm3 = (23.497*$beta0**2+6.248*$beta1+1.019*$beta0-29.94); $Zm3 = $Cm3*$as**3;
$Zm = 1 + $Zm1 + $Zm2 + $Zm3;
print "# Zm to pole mass (1,2,3 loop) 1 + $Zm1  $Zm2  $Zm3 = $Zm\n";

$M = $Zm*$mbar;
print "# pole mass = $M\n";

# HQET -> QCD matching
$as = $alpha/$pi;

$L = log($mu**2/$M**2);
$Lbar = log($mu**2/$mbar**2);
print "$L $Lbar\n";

$ln2 = log(2.0);
$a4 = 0.5174790617; # need to input a number Li4(1/2)

$nl = 3; # number of light flavors
$nm = 0; # number of charm flavors
$nh = 0; # number of heavy flavors
# 1-loop
$C1 = - 2/3 - $Lbar/2;
$C1Lbar = -1/2;

# 2-loop
$CG = - 177/64 - 5/72*$pi**2 - 1/18*$pi**2*$ln2 - 11/36*$zeta3
    + (-79/144 - 7/108*$pi**2)*$Lbar + 13/16*$Lbar**2;
$CH = 727/432 - 1/6*$pi**2;
$CL = 47/288 + 1/36*$pi**2 + 5/72*$Lbar - 1/24*$Lbar**2;
$CM = 0;
$C2Lbar = (-79/144 - 7/108*$pi**2) + 5/72;
$C2Lbar2 =  13/16 - 1/24;

$C2 = $CG + $CH*$nh + $CL*$nl + $CM*$nm;
$C2bar = $C1Lbar*(-2*$Cm1);
print "$CG, $Lbar, $zeta3, $pi \n";

#print "$CG, $CL, $C2Lbar, $C2, $C2bar \n";


# 3-loop
$CGG = -62575/62208 - 231253/46656*$pi**2 - 517/324*$pi**2*$ln2
    + 20/81*$pi**2*$ln2**2 + 5645/1296*$zeta3
    + 2089/486*$pi**2*$zeta3 - 17347/58320*$pi**4 - 49435/2592*$zeta5
    + 11/54*$ln2**4 + 44/9*$a4
    + (115/54 - 121/648*$pi**2 + 1/36*$pi**2*$ln2 + 37/48*$zeta3
       - 95/1944*$pi**4)*$Lbar
    + (2257/576 + 91/432*$pi**2)*$Lbar**2 - 13/8*$Lbar**3;
$CGH = 2051/96 - 24583/2430*$pi**2 + 361/27*$pi**2*$ln2 + 10/81*$pi**2*$ln2**2
    - 45869/5184*$zeta3 + 53/96*$pi**2*$zeta3
    - 1/20*$pi**4 - 85/32*$zeta5 - 10/81*$ln2**4 - 80/27*$a4
    + (-727/864 + 1/12*$pi**2)*$Lbar;
$CGL = 24457/46656 + 5575/8748*$pi**2 + 19/324*$pi**2*$ln2 - 1/81*$pi**2*$ln2**2
    + 3181/1944*$zeta3 - 379/116640*$pi**4
    - 1/162*$ln2**4 - 4/27*$a4
    + (-319/5184 + 11/972*$pi**2 + 83/216*$zeta3)*$Lbar
    + (-469/864 - 7/648*$pi**2)*$Lbar**2 + 25/144*$Lbar**3;
$CHH = - 5857/7776 + 1/405*$pi**2 + 11/18*$zeta3;
$CHL = - 193/432 + 29/648*$pi**2;
$CLL = 1751/46656 - 13/648*$pi**2 - 7/108*$zeta3 + 35/2592*$L + 5/432*$L**2
    - 1/216*$L**3;
$CGM = 0; $CHM = 0; $CLM = 0; $CMM = 0;
$C3 = $CGG + $CGH*$nh + $CGL*$nl + $CHH*$nh**2 + $CHL*$nh*$nl + $CLL*$nl**2
    + $CGM*$nm + $CHM*$nh*$nm + $CLM*$nl*$nm + $CMM*$nm**2;
$C3bar = $C1Lbar*(-2*$Cm2+$Cm1**2)
    + $C2Lbar*(-2*$Cm1) + $C2Lbar2*$Lbar*(-2*$Cm1);

# total
$Cmu1 = $as*$C1;
$Cmu2 = $as**2*($C2+$C2bar);
$Cmu3 = $as**3*($C3+$C3bar+$C1*(-$beta0*2*$Cm1));
$Cmu = 1 + $Cmu1 + $Cmu2 + $Cmu3;
print "C(mu)  1 + $Cmu1  $Cmu2  $Cmu3 = $Cmu\n";
