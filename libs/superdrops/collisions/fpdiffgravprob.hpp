/*
 * Copyright (c) 2024 MPI-M, Clara Bayley
 *
 *
 * ----- CLEO -----
 * File: fpdiffgravprob.hpp
 * Project: collisions
 * Created Date: Thursday 9th November 2023
 * Author: Clara Bayley (CB)
 * Additional Contributors:
 * -----
 * License: BSD 3-Clause "New" or "Revised" License
 * https://opensource.org/licenses/BSD-3-Clause
 * -----
 * File Description:
 * Header file for calculation of probability of a
 * collision-coalescence event between two (real)
 * droplets using the Golovin Kernel. Probability
 * calculations are contained in structures
 * that satisfy the requirements of the
 * PairProbability concept (see collisions.hpp)
 */

#ifndef LIBS_SUPERDROPS_COLLISIONS_FPDIFFGRAVPROB_HPP_
#define LIBS_SUPERDROPS_COLLISIONS_FPDIFFGRAVPROB_HPP_

#include <Kokkos_Core.hpp>

#include "../../cleoconstants.hpp"
#include "../superdrop.hpp"
#include "../terminalvelocity.hpp"

namespace dlc = dimless_constants;

/* Probability of collision-coalescence of
a pair of droplets according to Golovin 1963
(see e.g. Shima et al. 2009) */
struct FPDiffGravProb {
 private:
  double prob_jk_const;
  FPTerminalVelocity terminalv;
  KOKKOS_FUNCTION
  double lnEmax(const double log_g_r) const;
  KOKKOS_FUNCTION
  double lnEmin(const double log_g_r) const;
  KOKKOS_FUNCTION
  double a1(const double log_g_r) const;
  KOKKOS_FUNCTION
  double b1(const double log_g_r) const;
  KOKKOS_FUNCTION
  double overdamped_efficiency(const double log_kn, const double log_g_r) const;
  KOKKOS_FUNCTION
  double a2(const double log_g_r) const;
  KOKKOS_FUNCTION
  double b2(const double log_g_r) const;
  KOKKOS_FUNCTION
  double x02(const double log_g_r) const;
  KOKKOS_FUNCTION
  double expitl(const double x) const;
  KOKKOS_FUNCTION
  double oseen_efficiency(const double kn, const double g_r) const;
  KOKKOS_FUNCTION
  double oseen_efficiency_nooverdamped(const double kn, const double g_r) const;
  KOKKOS_FUNCTION
  double fdiff1(const double g_r) const;
  KOKKOS_FUNCTION
  double fdiff2(const double g_r) const;
  KOKKOS_FUNCTION
  double deltaE_diff_only(const double g_r, const double kn) const;
  KOKKOS_FUNCTION
  double fovdwa(const double g_r) const;
  KOKKOS_FUNCTION
  double fovdwb(const double g_r) const;
  KOKKOS_FUNCTION
  double overdamped_efficiency_vdw(const double kn, const double g_r) const;
  KOKKOS_FUNCTION
  double oseen_efficiency_vdw(const double kn, const double g_r) const;
  KOKKOS_FUNCTION
  double a_g(const double g_r) const;
  KOKKOS_FUNCTION
  double b_g(const double g_r) const;
  KOKKOS_FUNCTION
  double deltae_pe(const double pec, const double g_r) const;
  KOKKOS_FUNCTION
  double qpe_interp(const double pec) const;
  KOKKOS_FUNCTION
  double pe(const double aaa_radius, const double g_r) const;
  KOKKOS_FUNCTION
  double peeq(const double aaa_radius) const;

 public:
  FPDiffGravProb() : prob_jk_const(Kokkos::numbers::pi * dlc::R0 * dlc::R0 * dlc::W0) {}

  /* returns probability that a pair of droplets collide
  (and coalesce or breakup etc.) according to the hydrodynamic,
  i.e. gravitational, collision kernel. Probability is given by
  prob_jk = K(drop1, drop2) * delta_t/delta_vol, (see Shima 2009 eqn 3)
  where the kernel, K(drop1, drop2) := eff * pi * (r1 + r2)^2 * |v1−v2|,
  given the efficiency factor eff = eff(drop1, drop2), for
  example as expressed in equation 11 of Simmel at al. 2002 for
  collision-coalescence */
  KOKKOS_FUNCTION
  double operator()(const Superdrop &drop1, const Superdrop &drop2, const double DELT,
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
    const auto vtx = double{terminalv(drop1)};
    const auto vty = double{terminalv(drop2)};
    const auto vdiff = double{Kokkos::abs(vtx - vty)};

    const auto radius_ratio = double{Kokkos::pow(Kokkos::fmax(x, y)
                                     / Kokkos::fmin(x, y), 1.0 / 3.0)};
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
};



#endif  // LIBS_SUPERDROPS_COLLISIONS_FPDIFFGRAVPROB_HPP_
