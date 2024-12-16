#!/bin/sh

set -e
set -u

#source ./environment

mkdir -p $Poleno_output
mkdir -p $Poleno_output/Legacy
mkdir -p $Log_dir

f=$@
Zip_to_process=$(basename $f)
Zip_date=`echo $Zip_to_process | cut -b 1-14`
Log_file=$Log_dir/$Zip_date.log

if [ -f $Poleno_output/${Zip_date::-1}_pollen.csv ]; then
	echo $(date +"%H:%M:%S") 'Poleno hourly zip' $Zip_to_process 'was processed before'
else
	echo $(date +"%H:%M:%S") 'Poleno hourly zip' $Zip_to_process 'has to be processed, processing'
	echo 'Poleno hourly zip:' $Zip_to_process > $Log_file

	# echo 'Preparing' $Zip_to_process >> $Log_file
	Temp_folder=/dev/shm/Poleno_Recognition_Oper_$Zip_date
	if [ -d "$Temp_folder" ]; then rm -Rf $Temp_folder; fi
	mkdir $Temp_folder

	threshold_percentage=75
	while true; do
	    disk_usage=$(df -P /dev/shm/ | awk 'NR==2 {print $5}' | tr -d '%')
	    if [ "$disk_usage" -gt "$threshold_percentage" ]; then
	        echo $(date +"%H:%M:%S") "Disk usage is above $threshold_percentage%, waiting..."
	        sleep 60  # Wait for 60 seconds before checking again
	    else
	        # echo $(date +"%H:%M:%S") "Disk space is now above $threshold_percentage%"
	        break
	    fi
	done

	unzip -q $Poleno_hourly_zips/$Zip_to_process -d $Temp_folder
	echo 'Time elapsed for' $Zip_to_process 'preparation:' $SECONDS 'seconds' >> $Log_file

	SECONDS=0
	EXIT_CODE='none'
	python3.7 $Poleno_scripts/Recognition_11_classes_operational.py $Temp_folder $Poleno_output $Zip_date >> $Log_file || EXIT_CODE=$?
	echo 'Errors: ' $EXIT_CODE >> $Log_file
	echo 'Time elapsed for' $Zip_to_process 'processing:' $SECONDS 'seconds' >> $Log_file

	SECONDS=0
	# echo 'Cleaning'
	rm -Rf $Temp_folder
	echo 'Time elapsed for cleaning:' $SECONDS 'seconds' >> $Log_file

	cat $Poleno_output/*pollen.csv > $Poleno_output/../$Recognition_Results_file
	sed -i '1 i\Year,Month,Day,Hour,Alnus,Artemisia,Betula,Corylus,Fraxinus,Picea,Pinus,Populus,Quercus,Salix,Mist' $Poleno_output/../$Recognition_Results_file

fi
