# no-brain cut from recognition scripts

import time, os, shutil, io, glob, csv, json, gzip
from datetime import datetime
from dateutil import tz
import pytz
from zipfile import ZipFile
from PIL import Image
import numpy as np
import PolenoRecognitionConfigs as PRC
import PolenoAdditionalConfigs as PAC
import pandas as pd
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import load_model
sys.stderr = stderr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

t = time.time()

# message
def processing_msg(iter,json_iter,zip_date_iter):
    print('Processing', iter+1, 'of', len(json_iter), 'events for', os.path.basename(zip_date_iter), end="", flush=True)

# JSON Validation
## collect all Keys in Json
def getKeys(json):
    keys = []
    for k in json.keys():
        keys.append(k)
        if type(json[k]) == dict:
            k = getKeys(json[k])
            keys.extend(k)
    return keys

## check if all default keys in sample keys
def checkKeys(sample, default):
    sample = getKeys(sample)
    default = getKeys(default)
    valid = True
    for k in default:
        if not k in sample: valid = False
    return valid

## validate json
def jsonValidation(json, json_default):
    valid = checkKeys(json, json_default)
    return valid

# Holo image validation
def imgValidation(json):
    try:
        sol = np.array([json['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['solidity'],
                        json['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['solidity']])
        area = np.array([json['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['area'],
                         json['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['area']])
        minorAxis = np.array([json['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['minor_axis_length'],
                              json['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['minor_axis_length']])
        majorAxis = np.array([json['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['major_axis_length'],
                              json['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['major_axis_length']])
        perimeter = np.array([json['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['perimeter'],
                              json['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['perimeter']])
        maxIntensity = np.array([json['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['max_intensity'],
                                 json['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['max_intensity']])
        minIntensity = np.array([json['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['min_intensity'],
                                 json['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['min_intensity']])
        meanIntensity = np.array([json['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['mean_intensity'],
                                  json['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['mean_intensity']])
        eccentricity = np.array([json['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['eccentricity'],
                                 json['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['eccentricity']])
        return True
    except:
        return False
    return True

### Start

zip_date = sys.argv[3]
json_names = [s for s in os.listdir(sys.argv[1]) if 'ev.json' in s]

threshold = 0.70                # if set below 0.5 the event might get counted 2+ times
classes = np.zeros(11)          # adjust for number of classes
size_classes = np.zeros(15)     # adjust for number of classes
unrecognised = np.zeros(1)      # number of unrecognised events

bins = [4,5,6,7,8,9,10,11,12,14,16,18,20,25,30,60]

sol_min = 0.9
sol_max = 1.1

df_stats = pd.DataFrame(
    columns=['total_cnt', 'valid_cnt', 'json_failCnt', 'holo_failCnt', 'imgProp_failCnt', 'imgProp_exceptionCnt', 'trig_failCnt',
             'others_failCnt'], index=[0])
df_stats.iloc[0] = np.zeros(len(df_stats.columns))

# print('Loading model')
# model_name = sys.argv[2] + '/../' + 'Poleno_2021_10_plus_mist_Classes_Model_v2.h5'
model_name = 'Poleno_2021_10_plus_mist_Classes_Model_v2.h5'
model = load_model(model_name)
print('Processing measurements')
 
# All_areas = np.zeros(len(json_names))

for i in range(len(json_names)):

    curent_event_size_classes = np.zeros(15)

    # processing_msg(i,json_names,zip_date)
    # print()

    json_path_name = sys.argv[1] + '/' + json_names[i]
    json_name = os.path.basename(json_path_name)

    # with ZipFile(zip_date) as z:
    #     zf = z.read(json_path_name)
    #     json_data = json.loads(zf.decode("utf-8"))

    df_stats['total_cnt'] += 1
    valid = True

    if not os.stat(json_path_name).st_size>0:
        df_stats['others_failCnt'] += 1
        #print(json_name, '<-- json size is zero')
        continue

    with open(json_path_name) as j:
        json_data = json.loads(j.read())
    
    valid = valid and jsonValidation(json_data, PRC.json_default)
    if not valid:
        df_stats['json_failCnt'] += 1
        #print(json_name, '<-- json is invalid')
        # print('; json is invalid:',json_name)
        continue

    if PRC.write_imgProperties:
        valid = valid and imgValidation(json_data)
        if not valid:
            df_stats['imgProp_failCnt'] += 1
            #print(json_name, '<-- images are invalid')
            # print('; images are invalid:',json_name)
            continue

    event_name = sys.argv[1] + '/' + json_names[i]


    cam1 = event_name[:-7] + 'ev.computed_data.holography.image_pairs.0.0.rec_mag.png'
    cam2 = event_name[:-7] + 'ev.computed_data.holography.image_pairs.0.1.rec_mag.png'

    #print(cam1)

    # if ((cam1 not in rcg_zip.namelist()) or (cam2 not in rcg_zip.namelist())):
    if (not os.path.isfile(cam1)) or (not os.path.isfile(cam2)):
        valid = not valid
        df_stats['imgProp_failCnt'] += 1
        # print('; no images:',json_name)

    if valid:
        #print(json_name, '<-- event is valid')
        
        img1 = Image.open(cam1)
        img2 = Image.open(cam2)

        try:
            if img1.mode in ['L', 'P']: img1, img2 = np.array(img1) / (2**8), np.array(img2)/ (2**8)
            if img1.mode in ['I', 'F', 'I;16', 'I;16B', 'I;16L', 'I;16S', 'I;16BS', 'I;16LS']: img1, img2 = np.array(img1) / (2**16), np.array(img2)/ (2**16)

            image1 = np.expand_dims(img1, -1)
            image2 = np.expand_dims(img2, -1)
            image1 = np.expand_dims(image1, 0)
            image2 = np.expand_dims(image2, 0)

            try:
                result = model.predict([image1, image2])
            except:
                df_stats['imgProp_exceptionCnt'] += 1
                continue

            result = model.predict([image1, image2])
            result = result.transpose()

            df_stats['valid_cnt'] += 1

            # print(result)

            recognition = [r >= threshold for r in result]
            recognition = [int(elem) for elem in recognition]
            recognition = np.asarray(recognition)

            # print('; Recognition:', recognition, end="", flush=True)
            
            ### SHOULD BE REWRITEN ###

            sol = np.array([json_data['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['solidity'],
                            json_data['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['solidity']])
            area = np.array([json_data['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['area'],
                             json_data['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['area']])
            minorAxis = np.array([json_data['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['minor_axis_length'],
                                  json_data['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['minor_axis_length']])
            majorAxis = np.array([json_data['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['major_axis_length'],
                                  json_data['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['major_axis_length']])
            perimeter = np.array([json_data['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['perimeter'],
                                  json_data['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['perimeter']])
            maxIntensity = np.array([json_data['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['max_intensity'],
                                     json_data['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['max_intensity']])
            minIntensity = np.array([json_data['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['min_intensity'],
                                     json_data['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['min_intensity']])
            meanIntensity = np.array([json_data['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['mean_intensity'],
                                      json_data['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['mean_intensity']])
            eccentricity = np.array([json_data['computed_data']['holography']['image_pairs'][0][0]['rec_mag_properties']['eccentricity'],
                                     json_data['computed_data']['holography']['image_pairs'][0][1]['rec_mag_properties']['eccentricity']])

            if recognition[0] == 1:  # alnus
                if (sol <= sol_min*PAC.alnus_sol_range[0]).any() or (sol > sol_max*PAC.alnus_sol_range[1]).any() or (area <= PAC.alnus_area_range[0]).any() or (area > PAC.alnus_area_range[1]).any() or (minorAxis <= PAC.alnus_minorAxis_range[0]).any() or (minorAxis > PAC.alnus_minorAxis_range[1]).any() or (majorAxis <= PAC.alnus_majorAxis_range[0]).any() or (majorAxis > PAC.alnus_majorAxis_range[1]).any() or (perimeter <= PAC.alnus_perimeter_range[0]).any() or (perimeter > PAC.alnus_perimeter_range[1]).any() or (maxIntensity <= PAC.alnus_maxIntensity_range[0]).any() or (maxIntensity > PAC.alnus_maxIntensity_range[1]).any() or (minIntensity <= PAC.alnus_minIntensity_range[0]).any() or (minIntensity > PAC.alnus_minIntensity_range[1]).any() or (minIntensity <= PAC.alnus_minIntensity_range[0]).any() or (minIntensity > PAC.alnus_minIntensity_range[1]).any() or (meanIntensity <= PAC.alnus_meanIntensity_range[0]).any() or (meanIntensity > PAC.alnus_meanIntensity_range[1]).any() or (eccentricity <= PAC.alnus_eccentricity_range[0]).any() or (eccentricity > PAC.alnus_eccentricity_range[1]).any():
                    recognition[0] = 0  # not alnus
                    unrecognised = unrecognised + 1
                    # print('', json_name, 'does not pass additional filter', end="", flush=True)

            if recognition[1] == 1:  # artemisia
                if (sol <= sol_min*PAC.artemisia_sol_range[0]).any() or (sol > sol_max*PAC.artemisia_sol_range[1]).any() or (area <= PAC.artemisia_area_range[0]).any() or (area > PAC.artemisia_area_range[1]).any() or (minorAxis <= PAC.artemisia_minorAxis_range[0]).any() or (minorAxis > PAC.artemisia_minorAxis_range[1]).any() or (majorAxis <= PAC.artemisia_majorAxis_range[0]).any() or (majorAxis > PAC.artemisia_majorAxis_range[1]).any() or (perimeter <= PAC.artemisia_perimeter_range[0]).any() or (perimeter > PAC.artemisia_perimeter_range[1]).any() or (maxIntensity <= PAC.artemisia_maxIntensity_range[0]).any() or (maxIntensity > PAC.artemisia_maxIntensity_range[1]).any() or (minIntensity <= PAC.artemisia_minIntensity_range[0]).any() or (minIntensity > PAC.artemisia_minIntensity_range[1]).any() or (minIntensity <= PAC.artemisia_minIntensity_range[0]).any() or (minIntensity > PAC.artemisia_minIntensity_range[1]).any() or (meanIntensity <= PAC.artemisia_meanIntensity_range[0]).any() or (meanIntensity > PAC.artemisia_meanIntensity_range[1]).any() or (eccentricity <= PAC.artemisia_eccentricity_range[0]).any() or (eccentricity > PAC.artemisia_eccentricity_range[1]).any():
                    recognition[1] = 0  # not artemisia
                    unrecognised = unrecognised + 1
                    # print('', json_name, 'does not pass additional filter', end="", flush=True)

            if recognition[2] == 1:  # betula
                if (sol <= sol_min*PAC.betula_sol_range[0]).any() or (sol > sol_max*PAC.betula_sol_range[1]).any() or (area <= PAC.betula_area_range[0]).any() or (area > PAC.betula_area_range[1]).any() or (minorAxis <= PAC.betula_minorAxis_range[0]).any() or (minorAxis > PAC.betula_minorAxis_range[1]).any() or (majorAxis <= PAC.betula_majorAxis_range[0]).any() or (majorAxis > PAC.betula_majorAxis_range[1]).any() or (perimeter <= PAC.betula_perimeter_range[0]).any() or (perimeter > PAC.betula_perimeter_range[1]).any() or (maxIntensity <= PAC.betula_maxIntensity_range[0]).any() or (maxIntensity > PAC.betula_maxIntensity_range[1]).any() or (minIntensity <= PAC.betula_minIntensity_range[0]).any() or (minIntensity > PAC.betula_minIntensity_range[1]).any() or (minIntensity <= PAC.betula_minIntensity_range[0]).any() or (minIntensity > PAC.betula_minIntensity_range[1]).any() or (meanIntensity <= PAC.betula_meanIntensity_range[0]).any() or (meanIntensity > PAC.betula_meanIntensity_range[1]).any() or (eccentricity <= PAC.betula_eccentricity_range[0]).any() or (eccentricity > PAC.betula_eccentricity_range[1]).any():
                    recognition[2] = 0  # not betula
                    unrecognised = unrecognised + 1
                    # print('', json_name, 'does not pass additional filter', end="", flush=True)

            if recognition[3] == 1:  # corylus
                if (sol <= sol_min*PAC.corylus_sol_range[0]).any() or (sol > sol_max*PAC.corylus_sol_range[1]).any() or (area <= PAC.corylus_area_range[0]).any() or (area > PAC.corylus_area_range[1]).any() or (minorAxis <= PAC.corylus_minorAxis_range[0]).any() or (minorAxis > PAC.corylus_minorAxis_range[1]).any() or (majorAxis <= PAC.corylus_majorAxis_range[0]).any() or (majorAxis > PAC.corylus_majorAxis_range[1]).any() or (perimeter <= PAC.corylus_perimeter_range[0]).any() or (perimeter > PAC.corylus_perimeter_range[1]).any() or (maxIntensity <= PAC.corylus_maxIntensity_range[0]).any() or (maxIntensity > PAC.corylus_maxIntensity_range[1]).any() or (minIntensity <= PAC.corylus_minIntensity_range[0]).any() or (minIntensity > PAC.corylus_minIntensity_range[1]).any() or (minIntensity <= PAC.corylus_minIntensity_range[0]).any() or (minIntensity > PAC.corylus_minIntensity_range[1]).any() or (meanIntensity <= PAC.corylus_meanIntensity_range[0]).any() or (meanIntensity > PAC.corylus_meanIntensity_range[1]).any() or (eccentricity <= PAC.corylus_eccentricity_range[0]).any() or (eccentricity > PAC.corylus_eccentricity_range[1]).any():
                    recognition[3] = 0  # not corylus
                    unrecognised = unrecognised + 1
                    # print('', json_name, 'does not pass additional filter', end="", flush=True)

            if recognition[4] == 1:  # fraxinus
                if (sol <= sol_min*PAC.fraxinus_sol_range[0]).any() or (sol > sol_max*PAC.fraxinus_sol_range[1]).any() or (area <= PAC.fraxinus_area_range[0]).any() or (area > PAC.fraxinus_area_range[1]).any() or (minorAxis <= PAC.fraxinus_minorAxis_range[0]).any() or (minorAxis > PAC.fraxinus_minorAxis_range[1]).any() or (majorAxis <= PAC.fraxinus_majorAxis_range[0]).any() or (majorAxis > PAC.fraxinus_majorAxis_range[1]).any() or (perimeter <= PAC.fraxinus_perimeter_range[0]).any() or (perimeter > PAC.fraxinus_perimeter_range[1]).any() or (maxIntensity <= PAC.fraxinus_maxIntensity_range[0]).any() or (maxIntensity > PAC.fraxinus_maxIntensity_range[1]).any() or (minIntensity <= PAC.fraxinus_minIntensity_range[0]).any() or (minIntensity > PAC.fraxinus_minIntensity_range[1]).any() or (minIntensity <= PAC.fraxinus_minIntensity_range[0]).any() or (minIntensity > PAC.fraxinus_minIntensity_range[1]).any() or (meanIntensity <= PAC.fraxinus_meanIntensity_range[0]).any() or (meanIntensity > PAC.fraxinus_meanIntensity_range[1]).any() or (eccentricity <= PAC.fraxinus_eccentricity_range[0]).any() or (eccentricity > PAC.fraxinus_eccentricity_range[1]).any():
                    recognition[4] = 0  # not fraxinus
                    unrecognised = unrecognised + 1
                    # print('', json_name, 'does not pass additional filter', end="", flush=True)

            # if recognition[5] == 1:  # picea
                ### TO ADD ###

            if recognition[6] == 1:  # pinus
                if (sol <= sol_min*PAC.pinus_sol_range[0]).any() or (sol > sol_max*PAC.pinus_sol_range[1]).any() or (area <= PAC.pinus_area_range[0]).any() or (area > PAC.pinus_area_range[1]).any() or (minorAxis <= PAC.pinus_minorAxis_range[0]).any() or (minorAxis > PAC.pinus_minorAxis_range[1]).any() or (majorAxis <= PAC.pinus_majorAxis_range[0]).any() or (majorAxis > PAC.pinus_majorAxis_range[1]).any() or (perimeter <= PAC.pinus_perimeter_range[0]).any() or (perimeter > PAC.pinus_perimeter_range[1]).any() or (maxIntensity <= PAC.pinus_maxIntensity_range[0]).any() or (maxIntensity > PAC.pinus_maxIntensity_range[1]).any() or (minIntensity <= PAC.pinus_minIntensity_range[0]).any() or (minIntensity > PAC.pinus_minIntensity_range[1]).any() or (minIntensity <= PAC.pinus_minIntensity_range[0]).any() or (minIntensity > PAC.pinus_minIntensity_range[1]).any() or (meanIntensity <= PAC.pinus_meanIntensity_range[0]).any() or (meanIntensity > PAC.pinus_meanIntensity_range[1]).any() or (eccentricity <= PAC.pinus_eccentricity_range[0]).any() or (eccentricity > PAC.pinus_eccentricity_range[1]).any():
                    recognition[6] = 0  # not pinus
                    unrecognised = unrecognised + 1
                    # print('', json_name, 'does not pass additional filter', end="", flush=True)

            if recognition[7] == 1:  # populus
                # ratio1 = json_data['rawData']['sipmData']['4']['corrChannels']['corrMag'][2] / \
                #          json_data['rawData']['sipmData']['4']['corrChannels']['corrMag'][5]
                # ratio2 = json_data['rawData']['sipmData']['4']['corrChannels']['corrMag'][3] / \
                #          json_data['rawData']['sipmData']['4']['corrChannels']['corrMag'][5]
                if (sol <= sol_min*PAC.populus_sol_range[0]).any() or (sol > sol_max*PAC.populus_sol_range[1]).any() or (area <= PAC.populus_area_range[0]).any() or (area > PAC.populus_area_range[1]).any() or (minorAxis <= PAC.populus_minorAxis_range[0]).any() or (minorAxis > PAC.populus_minorAxis_range[1]).any() or (majorAxis <= PAC.populus_majorAxis_range[0]).any() or (majorAxis > PAC.populus_majorAxis_range[1]).any() or (perimeter <= PAC.populus_perimeter_range[0]).any() or (perimeter > PAC.populus_perimeter_range[1]).any() or (maxIntensity <= PAC.populus_maxIntensity_range[0]).any() or (maxIntensity > PAC.populus_maxIntensity_range[1]).any() or (minIntensity <= PAC.populus_minIntensity_range[0]).any() or (minIntensity > PAC.populus_minIntensity_range[1]).any() or (minIntensity <= PAC.populus_minIntensity_range[0]).any() or (minIntensity > PAC.populus_minIntensity_range[1]).any() or (meanIntensity <= PAC.populus_meanIntensity_range[0]).any() or (meanIntensity > PAC.populus_meanIntensity_range[1]).any() or (eccentricity <= PAC.populus_eccentricity_range[0]).any() or (eccentricity > PAC.populus_eccentricity_range[1]).any():
                    recognition[7] = 0  # not populus
                    unrecognised = unrecognised + 1
                    # print('', json_name, 'does not pass additional filter', end="", flush=True)
                # elif ratio1 < 10 or ratio2 < 5:
                #     # print(json_name, 'does not pass fluorescence check')
                #     recognition[10] = 0  # not populus
                #     unrecognised = unrecognised + 1

            if recognition[8] == 1:  # quercus
                if (sol <= sol_min*PAC.quercus_sol_range[0]).any() or (sol > sol_max*PAC.quercus_sol_range[1]).any() or (area <= PAC.quercus_area_range[0]).any() or (area > PAC.quercus_area_range[1]).any() or (minorAxis <= PAC.quercus_minorAxis_range[0]).any() or (minorAxis > PAC.quercus_minorAxis_range[1]).any() or (majorAxis <= PAC.quercus_majorAxis_range[0]).any() or (majorAxis > PAC.quercus_majorAxis_range[1]).any() or (perimeter <= PAC.quercus_perimeter_range[0]).any() or (perimeter > PAC.quercus_perimeter_range[1]).any() or (maxIntensity <= PAC.quercus_maxIntensity_range[0]).any() or (maxIntensity > PAC.quercus_maxIntensity_range[1]).any() or (minIntensity <= PAC.quercus_minIntensity_range[0]).any() or (minIntensity > PAC.quercus_minIntensity_range[1]).any() or (minIntensity <= PAC.quercus_minIntensity_range[0]).any() or (minIntensity > PAC.quercus_minIntensity_range[1]).any() or (meanIntensity <= PAC.quercus_meanIntensity_range[0]).any() or (meanIntensity > PAC.quercus_meanIntensity_range[1]).any() or (eccentricity <= PAC.quercus_eccentricity_range[0]).any() or (eccentricity > PAC.quercus_eccentricity_range[1]).any():
                    recognition[8] = 0  # not quercus
                    unrecognised = unrecognised + 1
                    # print('', json_name, 'does not pass additional filter', end="", flush=True)

            if recognition[9] == 1:  # salix
                # ratio1 = json_data['rawData']['sipmData']['4']['corrChannels']['corrMag'][2] / \
                #          json_data['rawData']['sipmData']['4']['corrChannels']['corrMag'][5]
                # ratio2 = json_data['rawData']['sipmData']['4']['corrChannels']['corrMag'][3] / \
                #          json_data['rawData']['sipmData']['4']['corrChannels']['corrMag'][5]
                # ratio3 = json_data['rawData']['sipmData']['4']['corrChannels']['corrMag'][4] / \
                #          json_data['rawData']['sipmData']['4']['corrChannels']['corrMag'][5]
                if (sol <= sol_min*PAC.salix_sol_range[0]).any() or (sol > sol_max*PAC.salix_sol_range[1]).any() or (area <= PAC.salix_area_range[0]).any() or (area > PAC.salix_area_range[1]).any() or (minorAxis <= PAC.salix_minorAxis_range[0]).any() or (minorAxis > PAC.salix_minorAxis_range[1]).any() or (majorAxis <= PAC.salix_majorAxis_range[0]).any() or (majorAxis > PAC.salix_majorAxis_range[1]).any() or (perimeter <= PAC.salix_perimeter_range[0]).any() or (perimeter > PAC.salix_perimeter_range[1]).any() or (maxIntensity <= PAC.salix_maxIntensity_range[0]).any() or (maxIntensity > PAC.salix_maxIntensity_range[1]).any() or (minIntensity <= PAC.salix_minIntensity_range[0]).any() or (minIntensity > PAC.salix_minIntensity_range[1]).any() or (minIntensity <= PAC.salix_minIntensity_range[0]).any() or (minIntensity > PAC.salix_minIntensity_range[1]).any() or (meanIntensity <= PAC.salix_meanIntensity_range[0]).any() or (meanIntensity > PAC.salix_meanIntensity_range[1]).any() or (eccentricity <= PAC.salix_eccentricity_range[0]).any() or (eccentricity > PAC.salix_eccentricity_range[1]).any():
                    recognition[9] = 0  # not salix
                    unrecognised = unrecognised + 1
                    # print('', json_name, 'does not pass additional filter', end="", flush=True)

    ###############################################################

                recognition[9] = 0  ### TO BE FIXED
                unrecognised = unrecognised + 1

    ###############################################################

            if recognition[10] == 1:  # mist
                if (sol <= sol_min*PAC.mist_sol_range[0]).any() or (sol > sol_max*PAC.mist_sol_range[1]).any() or (area <= PAC.mist_area_range[0]).any() or (area > PAC.mist_area_range[1]).any() or (minorAxis <= PAC.mist_minorAxis_range[0]).any() or (minorAxis > PAC.mist_minorAxis_range[1]).any() or (majorAxis <= PAC.mist_majorAxis_range[0]).any() or (majorAxis > PAC.mist_majorAxis_range[1]).any() or (perimeter <= PAC.mist_perimeter_range[0]).any() or (perimeter > PAC.mist_perimeter_range[1]).any() or (maxIntensity <= PAC.mist_maxIntensity_range[0]).any() or (maxIntensity > PAC.mist_maxIntensity_range[1]).any() or (minIntensity <= PAC.mist_minIntensity_range[0]).any() or (minIntensity > PAC.mist_minIntensity_range[1]).any() or (minIntensity <= PAC.mist_minIntensity_range[0]).any() or (minIntensity > PAC.mist_minIntensity_range[1]).any() or (meanIntensity <= PAC.mist_meanIntensity_range[0]).any() or (meanIntensity > PAC.mist_meanIntensity_range[1]).any() or (eccentricity <= PAC.mist_eccentricity_range[0]).any() or (eccentricity > PAC.mist_eccentricity_range[1]).any():
                    recognition[10] = 0  # not mist
                    unrecognised = unrecognised + 1
                    # print('', json_name, 'does not pass additional filter', end="", flush=True)
        except:
            recognition = np.zeros(11)
            recognition = np.int8(recognition)
            unrecognised = unrecognised + 1

        size = np.around(np.mean(majorAxis)/17*8, decimals=2)
        # print('; Mean MinAxis:', size, 'um', end="", flush=True)

        # print()

        # print(recognition)

        event_multiplier = json_data['event_multiplier']
        recognition = recognition*event_multiplier

        classes = classes + recognition
        if all(r < threshold for r in result):
            unrecognised = unrecognised + 1

        for j in range(0,15):
            if size >= bins[j] and size < bins[j+1]:
                curent_event_size_classes[j] = 1*event_multiplier

        size_classes = size_classes + curent_event_size_classes

print('Recognition results:', classes)

headers = [     'Year',
                'Month',
                'Day',
                'Hour',
                "Alnus",
                "Artemisia",
                "Betula",
                "Corylus",
                "Fraxinus",
                "Picea",
                "Pinus",
                "Populus",
                "Quercus",
                "Salix",
                "Mist"#,
                # 'Unrecognised',
                # 'Valid',
                # 'Total'
                # 'json_fail',
                # 'imgProp_fail'
                ]

to_csv_hours = np.array(['YYYY','MM','DD','HH'])
to_csv_data = np.zeros(11)
concentrator_coeff = 1.7
grains_to_m3 = 1/2.4

### UTC to local Helsinki time

helsinki_timezone = pytz.timezone('Europe/Helsinki')
utc_timezone = pytz.timezone('UTC')
utc_time = datetime.strptime(os.path.basename(zip_date)[0:13], '%Y-%m-%d_%H')
helsinki_time = utc_time.astimezone(helsinki_timezone)
to_csv_hours[0] = helsinki_time.strftime("%Y")
to_csv_hours[1] = helsinki_time.strftime("%m")
to_csv_hours[2] = helsinki_time.strftime("%d")
to_csv_hours[3] = helsinki_time.strftime("%H")

classes = classes.astype(float)
classes = classes*grains_to_m3*concentrator_coeff
classes = classes.round(decimals=2)
#to_csv_data[0:11] = classes[0:11]

# CM corrections and SFs; placeholder
tmp = 2 * classes[0] - 8 * classes[2]
to_csv_data[0] = np.maximum(tmp, 0)
to_csv_data[1] = classes[1] / 10
tmp = 8 * classes[2] - 2 * classes[0]
to_csv_data[2] = np.maximum(tmp, 0)
tmp = 2 * classes[3] - 8 * classes[2]
to_csv_data[3] = np.maximum(tmp, 0)
to_csv_data[4] = classes[4] * 2
to_csv_data[5] = classes[5] * 2
to_csv_data[6] = classes[6] * 4
to_csv_data[7] = classes[7] / 10
to_csv_data[8] = classes[8] * 2
to_csv_data[9] = classes[9] / 10
to_csv_data[10] = classes[10] * 2
to_csv_data = np.round(to_csv_data, 2)

# unrecognised = unrecognised.astype(float)
# unrecognised = unrecognised*grains_to_m3*concentrator_coeff
# unrecognised = unrecognised.round(decimals=2)
# to_csv_data[11] = unrecognised[0]
# totals = df_stats['total_cnt']*grains_to_m3*concentrator_coeff
# totals = totals.round(decimals=2)
# to_csv_data[12] = totals
pd_hours = pd.DataFrame({'hours':to_csv_hours})
pd_data = pd.DataFrame({'data':to_csv_data})
pd_to_csv = pd.concat([pd_hours.hours, pd_data.data], axis=0)
pd_to_csv = pd.DataFrame(pd_to_csv).T

rcg_log = sys.argv[2] + '/' + os.path.basename(zip_date)[0:13] + '_pollen.csv'
if os.path.isfile(rcg_log):
    local_time = time.localtime()
    current_time = time.strftime("%d_%b_%Y_%H_%M_%S", local_time)
    rcg_csv_legacy = sys.argv[2] + '/Legacy/' + os.path.basename(zip_date)[0:13] + '_' + current_time + '_pollen.csv'
    os.rename(rcg_log,  rcg_csv_legacy)

with open(rcg_log, 'w', newline='') as create_csv:
    # csv_creator = csv.writer(create_csv)
    pd_to_csv.to_csv(rcg_log, mode='a', index=False, header=False)

elapsed = time.time() - t
print('Elapsed time:', '{0:.1f}'.format(elapsed), 'seconds')
