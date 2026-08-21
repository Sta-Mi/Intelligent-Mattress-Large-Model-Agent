#!/usr/bin/env bash
set -euo pipefail

# BodyPressureSD's pressure/pose pickles only: depth images are not required by
# PressurePoseTransformer. Harvard Dataverse serves these files without a zip
# password. An optional first argument selects the destination data root.
root="${1:-BodyPressureSD}"
mkdir -p "$root/synth"

download() {
  local id="$1" name="$2"
  curl --fail --location --retry 3 \
    "https://dataverse.harvard.edu/api/access/datafile/$id" \
    --output "$root/synth/$name"
}

download 4642064 train_slp_lay_f_1to40_8549.p
download 4642082 train_slp_lay_f_41to70_6608.p
download 4642090 train_slp_lay_f_71to80_2184.p
download 4642067 train_slp_lay_m_1to40_8493.p
download 4642073 train_slp_lay_m_41to70_6597.p
download 4642069 train_slp_lay_m_71to80_2188.p
download 4642070 train_slp_lside_f_1to40_8136.p
download 4642076 train_slp_lside_f_41to70_6158.p
download 4642072 train_slp_lside_f_71to80_2058.p
download 4642063 train_slp_lside_m_1to40_7761.p
download 4642083 train_slp_lside_m_41to70_5935.p
download 4642071 train_slp_lside_m_71to80_2002.p
download 4642085 train_slp_rside_f_1to40_7677.p
download 4642079 train_slp_rside_f_41to70_6006.p
download 4642080 train_slp_rside_f_71to80_2010.p
download 4642068 train_slp_rside_m_1to40_7377.p
download 4642092 train_slp_rside_m_41to70_5817.p
download 4642081 train_slp_rside_m_71to80_1939.p
