#!/bin/sh

set -e
set -u

source ./environment

echo 'Processing zips'

printf '%s\0' $Poleno_hourly_zips/2024-06-15_*h.zip | xargs -0 -P 4 -n 1 bash \
 	$Poleno_scripts/Recognition_Oper_Processing.sh

# find $Poleno_hourly_zips/*.zip -mtime -25 -printf '%p\0' \
#  | xargs -0 -P 6 -n 1 sh $HOME/Poleno_scripts/Oper_Recognition_2022/Recognition_Oper_Processing.sh

echo 'Finished'
