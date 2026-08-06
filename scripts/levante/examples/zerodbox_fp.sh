#!/bin/bash
#SBATCH --job-name=fp2024
#SBATCH --partition=shared
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=florian.poydenot@mpimet.mpg.de
#SBATCH --mail-type=ALL
#SBATCH --account=um1487
#SBATCH --output=./zerodbox_fp_out.%j.out
#SBATCH --error=./zerodbox_fp_err.%j.out

### ---------------------------------------------------- ###
### ------------------ Input Parameters ---------------- ###
### ------ You MUST edit these lines to set your ------- ###
### ---- build type, directories, the executable(s) ---- ###
### -------- to compile, and your python script -------- ###
### ---------------------------------------------------- ###
do_build="true"
buildtype="openmp"
compilername="intel"
path2CLEO=${HOME}/CLEO/
path2build=/work/um1487/m301163/cleo_runs/zerodbox_ndrops/build_colls0d_2_24/zerodbox_fp/
build_flags="-DCLEO_COUPLED_DYNAMICS=null -DCLEO_DOMAIN=cartesian \
  -DCLEO_NO_ROUGHPAPER=true -DCLEO_NO_PYBINDINGS=true"
executables="flocolls"

pythonscript=${path2CLEO}/examples/zerodbox_fp/zerodbox_fp.py
src_config_filename=${path2CLEO}/examples/zerodbox_fp/src/config/zerodbox_fp_config.yaml
script_args="${src_config_filename} --kernels diffgrav \
  --do_inputfiles --do_run_executable --do_plot_results"
### ---------------------------------------------------- ###
### ---------------------------------------------------- ###
### ---------------------------------------------------- ###

### ---------- build, compile and run example ---------- ###
${path2CLEO}/scripts/levante/examples/build_compile_run_plot.sh ${do_build} \
  ${buildtype} ${compilername} ${path2CLEO} ${path2build} "${build_flags}" \
  "${executables}" ${pythonscript} "${script_args}"
### ---------------------------------------------------- ###
