"""
    Swisens Poleno Recognition Configs module
    ------------
    This module is for setting configurations for the cleaning process with the Poleno_recognition_sorting.py module.

"""

# CONFIGS

# image validation

## Main validation parameters (0,float("inf"))
area_range=(100,30000)
sol_range=(0.7,1.0)
eccentricity_range=(0.1,1.0)

## Additional image validation options
minorAxis_range=(10,300)
majorAxis_range=(10,500)
perimeter_range=(20,1000)
maxIntensity_range=(0,float("inf"))
minIntensity_range=(0,float("inf"))
meanIntensity_range=(0,float("inf"))


# Clean by Trigger

## Enable/disable cleaning by trigger
cleanTrigger = False
display=False

## find peaks
peak_width=5
peak_prominence=2e3
peak_distance=50
peak_minDeviation=-10e3
peak_number=2

## Low level (no peak)
lowLevel_indOffset=100
lowLevelOffset_start=3e3  
lowLevelOffset_end=7e3
## First and second trigger peak must be in index range
range_first=(600,830)
range_second=(950,1100)


# Validation of the JSON format

## append image properties to the measurement json
write_imgProperties = True

# ## Load only zipped JSON files
# ending = '.json.gz'

## Load JSON files
ending = '.json'

## Default json file -> If the json file has at least these keys, it will be accepted
json_default = {
    "valid": True,
    "trigger_peak_delay": 0.0,
    "computed_data":[],
    "hw_timestamp":0.0,
#    "holo0": {
#        "xy": [0.0,0.0],
#        "zRough": 0.0,
#        "zFine": 0.0
#    },
#    "holo1": {
#        "xy": [0.0,0.0],
#        "zRough": 0.0,
#        "zFine": 0.0
#    },
    "velocity": 0.0,
#    "adcDump": {
#        "0A": [],
#        "0B": [],
#        "1A": [],
#        "1B": [],
#        "2A": [],
#        "2B": []
#    }
}
