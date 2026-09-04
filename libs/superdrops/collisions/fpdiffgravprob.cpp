/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: fpdiffgravprob.cpp
 * Project: collisions
 * Created Date: Thursday 9th November 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * functionality to calculate the probability of a
 * collision-coalescence event between two (real)
 * droplets using the Golovin Kernel. Probability
 * calculations are contained in structures
 * that satisfy the requirements of the
 * PairProbability concept (see collisions.hpp)
 */

#include "./fpdiffgravprob.hpp"


/*********************
 * EFFICIENCY
 *********************/

KOKKOS_FUNCTION
double FPDiffGravProb::lnEmax(const double log_g_r) const {
  return double{(1.0
                 - Kokkos::exp(-0.4419871 * log_g_r)) * (0.08168582 - 1.76428132 * log_g_r)
                 - 1.81053907};
}

KOKKOS_FUNCTION
double FPDiffGravProb::lnEmin(const double log_g_r) const {
  return double{(0.99981622 - 0.42894397 * Kokkos::exp(-0.3371372 * log_g_r * log_g_r))
                *(-19.47919739 - 2.19815948 * log_g_r)};
}

KOKKOS_FUNCTION
double FPDiffGravProb::a1(const double log_g_r) const {
  return double{0.22840486 - 0.01235286 * Kokkos::exp(-1.06857073 * log_g_r)};
}

KOKKOS_FUNCTION
double FPDiffGravProb::b1(const double log_g_r) const {
  return double{1.0881953 - 0.09722003 * Kokkos::exp(-1.17055577 * log_g_r)};
}

KOKKOS_FUNCTION
double FPDiffGravProb::overdamped_efficiency(const double log_kn, const double log_g_r) const {
  const auto lEmin = lnEmin(log_g_r);
  const auto lEmax = lnEmax(log_g_r);
  const auto lE = double{0.5 * (lEmax - lEmin) * Kokkos::tanh(a1(log_g_r) * log_kn + b1(log_g_r))
                         + 0.5 * (lEmax + lEmin)};
  return Kokkos::exp(lE);
}

KOKKOS_FUNCTION
double FPDiffGravProb::a2(const double log_g_r) const {
  return double{5.99492538e-02 * Kokkos::exp(-2.24769147 * log_g_r)
         + 1.20687911e-03 * log_g_r * log_g_r + 1.0};
}

KOKKOS_FUNCTION
double FPDiffGravProb::b2(const double log_g_r) const {
  return double{4.94224382 - 0.22235347*log_g_r};
}

KOKKOS_FUNCTION
double FPDiffGravProb::x02(const double log_g_r) const {
  return double{-8.53167718 * (-0.14559349 * log_g_r + 1.0)
                     * (1.0 - 0.37759413 * Kokkos::exp(-0.2120766 * log_g_r))};
}

KOKKOS_FUNCTION
double FPDiffGravProb::expitl(const double x) const {
  return double{1.0 / (1.0 + Kokkos::exp(-x))};
}

KOKKOS_FUNCTION
double FPDiffGravProb::oseen_efficiency(const double kn, const double g_r) const {
  const auto log_kn = Kokkos::log(kn);
  const auto log_g_r = Kokkos::log(g_r);
  return overdamped_efficiency(log_kn, log_g_r)
         + a2(log_g_r) * expitl(-b2(log_g_r) * (log_kn - x02(log_g_r)));
}

KOKKOS_FUNCTION
double FPDiffGravProb::oseen_efficiency_nooverdamped(const double kn, const double g_r) const {
  const auto log_kn = Kokkos::log(kn);
  const auto log_g_r = Kokkos::log(g_r);
  return a2(log_g_r) * expitl(-b2(log_g_r) * (log_kn - x02(log_g_r)));
}

KOKKOS_FUNCTION
double FPDiffGravProb::fdiff1(const double g_r) const {
  return double{-3.70416264 * Kokkos::exp(-g_r * 0.01221076)};
}

KOKKOS_FUNCTION
double FPDiffGravProb::fdiff2(const double g_r) const {
  return double{6.51924695 * Kokkos::exp(-g_r * 0.13578872)};
}

KOKKOS_FUNCTION
double FPDiffGravProb::deltaE_diff_only(const double kn, const double g_r) const {
  return Kokkos::exp(fdiff1(g_r) / kn + fdiff2(g_r));
}

KOKKOS_FUNCTION
double FPDiffGravProb::fovdwa(const double g_r) const {
  return double{17.64328547 / (Kokkos::pow(Kokkos::log(g_r), 5) + 93.30412203) - 1.69646329};
}

KOKKOS_FUNCTION
double FPDiffGravProb::fovdwb(const double g_r) const {
  return double{-0.90759351 * Kokkos::log(g_r) + 2.02800579 - 0.99066826 / Kokkos::sqrt(g_r)};
}

KOKKOS_FUNCTION
double FPDiffGravProb::overdamped_efficiency_vdw(const double kn, const double g_r) const {
  const auto a = double{1.0/kn};
  return Kokkos::pow(10.0, (fovdwa(g_r) * Kokkos::log10(a) + fovdwb(g_r)));
}

KOKKOS_FUNCTION
double FPDiffGravProb::oseen_efficiency_vdw(const double kn, const double g_r) const {
  return oseen_efficiency_nooverdamped(kn, g_r)+overdamped_efficiency_vdw(kn, g_r);
}

KOKKOS_FUNCTION
double FPDiffGravProb::a_g(const double g_r) const {
  return double{-0.8 + 0.4 / Kokkos::sqrt(g_r)};
}

KOKKOS_FUNCTION
double FPDiffGravProb::b_g(const double g_r) const {
  return double{0.10870377 * Kokkos::log(g_r) + 0.13430814};
}

KOKKOS_FUNCTION
double FPDiffGravProb::deltae_pe(const double pec, const double g_r) const {
  return Kokkos::pow(10.0, (a_g(g_r) * Kokkos::log10(pec) + b_g(g_r)));
}

KOKKOS_FUNCTION
double FPDiffGravProb::qpe_interp(const double pec) const {
  constexpr double a = 0.86915474;
  constexpr double b = 0.70032847;
  return double{1.0 + 4.0 / pec + ((1.0 - Kokkos::tanh(a * (Kokkos::log10(pec) - b)))/2.0)};
}

KOKKOS_FUNCTION
double FPDiffGravProb::terminalv(const double x) const {
  constexpr double inu_14 = 3.0e4;                       // Supmat PRF [2] Eq. (7) inu^(1/4)
  constexpr double igamma = 2.8e22;                      // Supmat PRF [2] Eq. (7)

  const auto c1 = double{Kokkos::pow(1.0 + x / igamma, 1.0 / 6.0)};
  const auto c3 = double{Kokkos::pow(x, 1.0 / 6.0)};
  const auto c2 = double{inu_14 / c3};  // order switch for optimization
  const auto c4 = double{(Kokkos::sqrt(c2 * c2 + 4.0 * c1 * c3) - c2)/(2.0 * c1)};

  return c4 * c4;
}

KOKKOS_FUNCTION
double FPDiffGravProb::pe(const double aaa_radius, const double g_r) const {
  double RR[2];
  double mm[2];
  double cd[2];
  double ut[2];
  double unstau[2];
  double tau[2];
  double ddPe[2];
  constexpr double gsacube = 9.00183E-6;
  constexpr double ddd = 830.0;
  constexpr double mathcalK = 0.04222572524923858;
  double ggg;
  int ii;
  RR[0] = double{1.0 + g_r};
  RR[1] = double{1.0 + 1.0 / g_r};

  ggg = gsacube * aaa_radius * aaa_radius * aaa_radius;

  for (ii = 0; ii <= 1; ii++) {
    mm[ii] = RR[ii] * RR[ii] * RR[ii];
    cd[ii] = double{3.0 * RR[ii] / (8.0 * ddd)};
    ut[ii] = double{(-1.0 + Kokkos::sqrt(1.0 + 8.0 * cd[ii] * ggg * RR[ii] * RR[ii] / 9.0))
                    / (2.0 * cd[ii])};
    unstau[ii] = double{4.5 * (1.0 + cd[ii] * ut[ii]) / (RR[ii] * RR[ii])};
    tau[ii] = double{1.0 / unstau[ii]};
    ddPe[ii] = mathcalK / aaa_radius / mm[ii] * tau[ii];
    ddPe[ii] = mathcalK / aaa_radius / mm[ii] * tau[ii];
  }
  return (RR[0] + RR[1]) * (ut[0] - ut[1]) / (ddPe[0] + ddPe[1]);
}

KOKKOS_FUNCTION
double FPDiffGravProb::peeq(const double aaa_radius) const {
  double RR[2];
  double mm[2];
  double cd[2];
  double ut[2];
  double unstau[2];
  double tau[2];
  double ddPe[2];
  constexpr double gsacube = 9.00183E-6;
  constexpr double ddd = 830.0;
  constexpr double mathcalK = 0.04222572524923858;
  double ggg;
  int ii;
  RR[0] = double{2.0};
  RR[1] = double{2.0};

  ggg = gsacube * aaa_radius * aaa_radius * aaa_radius;

  for (ii = 0; ii <= 1; ii++) {
    mm[ii] = RR[ii] * RR[ii] * RR[ii];
    cd[ii] = double{3.0 * RR[ii] / (8.0 * ddd)};
    ut[ii] = double{(-1.0 + Kokkos::sqrt(1.0 + 8.0 * cd[ii] * ggg * RR[ii] * RR[ii] / 9.0))
                    / (2.0 * cd[ii])};
    unstau[ii] = double{4.5 * (1.0 + cd[ii] * ut[ii]) / (RR[ii] * RR[ii])};
    tau[ii] = double{1.0 / unstau[ii]};
    ddPe[ii] = mathcalK / aaa_radius / mm[ii] * tau[ii];
    ddPe[ii] = mathcalK / aaa_radius / mm[ii] * tau[ii];
  }
  return (RR[0] + RR[1]) * (ut[0]) / (ddPe[0] + ddPe[1]);
}


/* returns probability that a pair of droplets coalesces
according to Golovin's (sum of volumes) coalescence kernel.
Prob equation is : prob_jk = K(drop1, drop2) * delta_t/delta_vol where
K(drop1, drop2) := C(drop1, drop2) * |v1−v2|, (see Shima 2009 eqn 3),
and K(drop1, drop2) is Golovin 1963 (coalescence) kernel */
KOKKOS_FUNCTION
double FPDiffGravProb::operator()(const Superdrop &drop1, const Superdrop &drop2, const double DELT,
                                  const double VOLUME) const {
  const auto DELT_DELVOL = double{DELT / VOLUME};

  constexpr double mathcalA = 1.92671441e-10 / dlc::R0;  // equivalent size A in units of R0
  constexpr double mathcalAcubed = mathcalA * mathcalA * mathcalA;
  constexpr double lbar = 352.93243;
  const auto x = double{drop1.rcubed() / mathcalAcubed};
  const auto y = double{drop2.rcubed() / mathcalAcubed};

  /* calculate Hydrodynamic Kernel*/
  const auto sumr = double{drop1.get_radius() + drop2.get_radius()};
  const auto sumrsqrd = double{sumr * sumr};
  const auto vtx = double{terminalv(x)};
  const auto vty = double{terminalv(y)};
  const auto vdiff = double{Kokkos::abs(vtx - vty)};

  const auto radius_ratio = double{Kokkos::pow(Kokkos::fmax(x, y) / Kokkos::fmin(x, y), 1.0 / 3.0)};
  const auto kn = double{lbar * (Kokkos::pow(x, 1.0 / 3.0)
                                 + Kokkos::pow(y, 1.0 / 3.0)) / Kokkos::pow(x * y, 1.0 / 3.0)};
  auto pecl = double{0.0};
  auto eff = double{0.0};
  auto diffgrav_kernel = double{0.0};

  if (x != y) {
    pecl = pe(1.0 / kn, radius_ratio);
    eff = double{4.0 / pecl + deltae_pe(pecl, radius_ratio)
                 + oseen_efficiency_vdw(kn, radius_ratio)};
    diffgrav_kernel = double{prob_jk_const * eff * sumrsqrd * vdiff};
  } else {
    pecl = peeq(1.0 / kn);
    eff = double{4.0 / pecl};
    diffgrav_kernel = double{prob_jk_const * eff * sumrsqrd * vtx};
  }

  /* calculate probability prob_jk analogous Shima 2009 eqn 3 */
  const auto prob_jk = diffgrav_kernel * DELT_DELVOL;

  return prob_jk;
}
