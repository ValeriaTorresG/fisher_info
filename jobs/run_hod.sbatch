#!/bin/bash
#SBATCH --job-name=astra_hod
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00:00
#SBATCH --mem=0
#SBATCH --array=0-0
#SBATCH --hint=nomultithread
#SBATCH --output=/pscratch/sd/v/vtorresg/hod-astra/logs/zone_%A_%a.out
#SBATCH --error=/pscratch/sd/v/vtorresg/hod-astra/logs/zone_%A_%a.err
#SBATCH --chdir=/global/homes/v/vtorresg/fisher_info

set -euo pipefail

module load python/3.12

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

export ASTRA_PROGRESS=1
export ASTRA_CLASS_SPLIT_ITER="${ASTRA_CLASS_SPLIT_ITER:-0}"
export ASTRA_CLASS_SKIP_COMBINED="${ASTRA_CLASS_SKIP_COMBINED:-0}"
export ASTRA_PROB_SPLIT_ITER="${ASTRA_PROB_SPLIT_ITER:-0}"
export ASTRA_PROB_SKIP_COMBINED="${ASTRA_PROB_SKIP_COMBINED:-0}"
export ASTRA_SIM_NRAND_FACTOR="${ASTRA_SIM_NRAND_FACTOR:-20}"
export PAIR_NJOBS_CAP="${PAIR_NJOBS_CAP:-32}"

R_LOWER="${R_LOWER:--0.25}"
R_MED="${R_MED:-0.25}"
R_UPPER="${R_UPPER:-0.65}"
N_RANDOM="${N_RANDOM:-100}"
WEBTYPE="${WEBTYPE:-filament}"
SOURCE="${SOURCE:-data}"
ZONE_OVERRIDE_DEFAULT="c000_ph000_seed0_hod000"

PROJECT_ROOT="$PWD"
MAIN_SCRIPT="${PROJECT_ROOT}/src/main.py"
LIST_SCRIPT="${PROJECT_ROOT}/src/utils/list_yuan23_snapshots.py"

BASE_DIR="${BASE_DIR:-/pscratch/sd/n/ntbfin/emulator/hods/z0.8/yuan23_prior}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/pscratch/sd/v/vtorresg/hod-astra}"

RAW_OUT="${SCRATCH_ROOT}/raw"
CLASS_ROOT="${SCRATCH_ROOT}/release"
GROUPS_OUT="${SCRATCH_ROOT}/groups"
PLOTS_OUT="${SCRATCH_ROOT}/figs"
LOGDIR="${SCRATCH_ROOT}/logs"
SPILL_DIR="${SCRATCH_ROOT}/tmp"
ZONE_LIST="${SCRATCH_ROOT}/zones_yuan23.txt"

export ASTRA_PAIR_SPILL_DIR="${ASTRA_PAIR_SPILL_DIR:-${SPILL_DIR}}"
export ASTRA_GROUP_SPILL_DIR="${ASTRA_GROUP_SPILL_DIR:-${SPILL_DIR}}"
export ASTRA_TMPDIR="${ASTRA_TMPDIR:-${SPILL_DIR}}"

if [[ -n ${SIM_LBOX:-} ]]; then
  export ASTRA_SIM_LBOX="${SIM_LBOX}"
fi

mkdir -p "${RAW_OUT}" "${CLASS_ROOT}" "${GROUPS_OUT}" "${PLOTS_OUT}" "${LOGDIR}" "${SPILL_DIR}"

tmp_zone_list="${ZONE_LIST}.tmp"
python "${LIST_SCRIPT}" --base-dir "${BASE_DIR}" --zones-only > "${tmp_zone_list}"
mv "${tmp_zone_list}" "${ZONE_LIST}"

TOTAL_ZONES="$(wc -l < "${ZONE_LIST}")"
if [[ "${TOTAL_ZONES}" -le 0 ]]; then
  echo "error: no zones discovered in ${BASE_DIR}" >&2
  exit 1
fi

if [[ -z ${ZONE_OVERRIDE:-} && -n ${ZONE_OVERRIDE_DEFAULT} ]]; then
  ZONE_OVERRIDE="${ZONE_OVERRIDE_DEFAULT}"
fi

if [[ -n ${ZONE_OVERRIDE:-} ]]; then
  ZONE_LABEL="${ZONE_OVERRIDE}"
  if ! grep -Fxq "${ZONE_LABEL}" "${ZONE_LIST}"; then
    echo "error: ZONE_OVERRIDE=${ZONE_LABEL} not found in ${ZONE_LIST}" >&2
    exit 1
  fi
else
  if (( SLURM_ARRAY_TASK_ID >= TOTAL_ZONES )); then
    echo "error: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} but TOTAL_ZONES=${TOTAL_ZONES}" >&2
    exit 1
  fi

  LINE_NO="$((SLURM_ARRAY_TASK_ID + 1))"
  ZONE_LABEL="$(sed -n "${LINE_NO}p" "${ZONE_LIST}")"
  if [[ -z "${ZONE_LABEL}" ]]; then
    echo "error: could not resolve zone label at line ${LINE_NO}" >&2
    exit 1
  fi
fi

TRACE_ARGS=()
if [[ -n ${OUT_TAG:-} ]]; then
  TRACE_ARGS+=(--out-tag "${OUT_TAG}")
fi
if [[ -n ${CHUNK_ROWS:-} ]]; then
  TRACE_ARGS+=(--chunk-rows "${CHUNK_ROWS}")
fi

LOG_PREFIX="${LOGDIR}/zone_${ZONE_LABEL}"
echo "[yuan23] zone=${ZONE_LABEL} n_random=${N_RANDOM} factor=${ASTRA_SIM_NRAND_FACTOR}" >&2

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores \
     /usr/bin/time -f $'%E elapsed\n%U user CPU\n%S sys CPU\n%P CPU%%\n%M KB max RSS\n' \
     -o "${LOG_PREFIX}.time" \
     python "${MAIN_SCRIPT}" \
       --release YUAN23 \
       --zones "${ZONE_LABEL}" \
       --base-dir "${BASE_DIR}" \
       --raw-out "${RAW_OUT}" \
       --class-out "${CLASS_ROOT}" \
       --groups-out "${GROUPS_OUT}" \
       --plot-output "${PLOTS_OUT}" \
       --spill-dir "${SPILL_DIR}" \
       --r-lower "${R_LOWER}" \
       --r-med "${R_MED}" \
       --r-upper "${R_UPPER}" \
       --webtype "${WEBTYPE}" \
       --source "${SOURCE}" \
       --n-random "${N_RANDOM}" \
       "${TRACE_ARGS[@]}" \
       --progress
