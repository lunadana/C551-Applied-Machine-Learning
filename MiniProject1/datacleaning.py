#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:10:18 2022
@author: lunadana

Mini Project 1
Data Cleaning

Links to dataset :
    - http://archive.ics.uci.edu/ml/datasets/Hepatitis
    - https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set#

Hepatitis : 
    7. Attribute information: 
     1. Class: DIE (1), LIVE(2) (Y)
     2. AGE: 10, 20, 30, 40, 50, 60, 70, 80
     3. SEX: male, female
     4. STEROID: no, yes
     5. ANTIVIRALS: no, yes
     6. FATIGUE: no, yes
     7. MALAISE: no, yes
     8. ANOREXIA: no, yes
     9. LIVER BIG: no, yes
    10. LIVER FIRM: no, yes
    11. SPLEEN PALPABLE: no, yes
    12. SPIDERS: no, yes
    13. ASCITES: no, yes
    14. VARICES: no, yes
    15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
        -- see the note below
    16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
    17. SGOT: 13, 100, 200, 300, 400, 500, 
    18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
    19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90
    20. HISTOLOGY: no, yes
    
Messidor : 
    
0) The binary result of quality assessment. 0 = bad quality 1 = sufficient quality.
    
1) The binary result of pre-screening, where 1 indicates severe retinal abnormality and 0 its lack.

2-7) The results of MA (Messidor Anomalie?) detection. Each feature value stand for the
number of MAs found at the confidence levels alpha = 0.5, . . . , 1, respectively.

8-15) contain the same information as 2-7) for exudates. However,
as exudates are represented by a set of points rather than the number of
pixels constructing the lesions, these features are normalized by dividing the
number of lesions with the diameter of the ROI to compensate different image
sizes. 

16) The euclidean distance of the center of
the macula and the center of the optic disc to provide important information
regarding the patientâ€™s condition. This feature
is also normalized with the diameter of the ROI.

17) The diameter of the optic disc.
    
18) The binary result of the AM/FM-based classification.
    
19) Class label. 1 = contains signs of DR (Accumulative label for the Messidor classes 1, 2, 3), 0 = no signs of DR.

"""

import pandas as pd 
import csv
import seaborn as sns

input_hepatitis = "/Users/lunadana/Desktop/COMP551/MiniProject/Data/hepatitis.data"
input_messidor = "/Users/lunadana/Desktop/COMP551/MiniProject/Data/messidor_features.arff"

#  ------------------------------ Hepatitis Data ------------------------------
df_hepatitis = pd.read_csv(input_hepatitis, header = None)

df_hepatitis.columns = ['Class','Age','Sex','Steroid','Antivirals','Fatigue','Malaise',
                        'Anorexia','LiverBig','LiverFirm','SpleenPalpable','Spiders','Ascites',
                        'Varices','Bilirubin','AlkPhosphate','Sgot','Albumin','Protime','Histology']

# Removing missing or malformed features 
df_hepatitis = df_hepatitis[~df_hepatitis.eq('?').any(1)]

#  ------------------------------ Messidor Data ------------------------------
data = []
with open(input_messidor, "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    i = 0
    for lines in csv_reader:
        i = i+1
        if i > 24:
            Data = data.append(lines)
            
df_messidor = pd.DataFrame(columns = ['Quality','Pre-screening','MAdetectionCL0.5','MAdetectionCL0.6',
                                      'MAdetectionCL0.7','MAdetectionCL0.8','MAdetectionCL0.9',
                                      'MAdetectionCL1','8','9','10','11','12','13','14','15',
                                      'EuclidianDistance','OpticDiscDiameter','AM/FM-basedClass','SignofDR'], data=data).astype(float) 
    

# ------------------------------ General Stat ------------------------------
df_messidor_stat = df_messidor.groupby('SignofDR').agg({'OpticDiscDiameter':'mean','MAdetectionCL0.5':'mean','MAdetectionCL1':'mean'})

sns.countplot(y = 'Pre-screening', hue = 'SignofDR', data = df_messidor)
df_messidor['SignofDR'].value_counts()

df_hepatitis.to_csv("hepatitis_clean.csv")
df_messidor.to_csv("messidor_clean.csv")





