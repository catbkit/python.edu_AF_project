import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

from pandas.tools import plotting
from sklearn import datasets, linear_model
from pandas.stats.api import ols
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    district, districtGeometry, districtType, districtRatings, districtAEA, districtDistinctions, districtReference, \
        districtECHS, districtStaffStudents, districtWealth, districtChapter41, districtChapter41Recapture, districtExpenditures \
        = readDistrictFiles()
    district = combineDistrictFiles(district, districtGeometry, districtType, districtRatings, districtAEA,
                                    districtDistinctions, districtReference, districtECHS, districtStaffStudents, 
                                    districtWealth, districtChapter41, districtChapter41Recapture, districtExpenditures)
    district = removeNotRated(district)
    district = fixString(district)
    district = fixMissing(district)
    district = logTransform(district)
    
    targetField = 'DOMAIN_IV_SCORE'
    
    district = createDummies(district, 'TYPE_DESCRIPTION')
    
    district = dropRowsMissingScores(district,targetField)
    
    toCSV(district)
    #plt.figure()
    #district['DI1'].diff().hist()
    #plt.figure()
    #district['DI2'].diff().hist()
    #plt.figure()
    #district['DI3'].diff().hist()
    #plt.figure()
    #district['DI4'].diff().hist()
    #plotting.scatter_matrix(district[['DI1','2016_CTE_PCT','2016_TEACHER_TURNOVER_RATIO','2016_ECODIS_PCT', '2015_OPERATING_EXPENDITURES_PER_STUDENT','2016_GT_PCT','2016_TEACHER_STUDENT_RATIO']])    

    # encode TYPE_DESCRIPTION as TYPE_DESCRIPTION_ORD
    district['DISTRICT_TYPE_NOM'] = pd.Categorical(district.DISTRICT_TYPE).codes
    district['CHARTER_OPERATOR_NOM'] = pd.Categorical(district.CHARTER_OPERATOR).codes
   
    y = district[targetField]
    X = district[[
        #'2016_CTE_PCT'
        '2016_GT_PCT'
        #'2016_ECODIS_PCT'
        ,'TYPE_DESCRIPTION_CHARTER_SCHOOL_DISTRICTS'
        ,'TYPE_DESCRIPTION_RURAL'
        ,'TYPE_DESCRIPTION_OTHER_CENTRAL_CITY'
        ,'TYPE_DESCRIPTION_NON-METROPOLITAN_FAST_GROWING'
        ,'TYPE_DESCRIPTION_MAJOR_SUBURBAN'
        ,'TYPE_DESCRIPTION_NON-METROPOLITAN_STABLE'
        ,'TYPE_DESCRIPTION_INDEPENDENT_TOWN'
        ,'TYPE_DESCRIPTION_OTHER_CENTRAL_CITY_SUBURBAN'
        ,'TYPE_DESCRIPTION_MAJOR_URBAN'
        #,'2015_MOBILITY_PCT'
        ,'2016_AT_RISK_PCT'
        ,'2016_LEP_PCT'
        ,'2016_SPED_PCT'
        #,'2016_TEACHER_TURNOVER_RATIO'
        #,'2016_TEACHER_TENURE_AVERAGE'
        #,'2016_TEACHER_EXPERIENCE_AVERAGE'
        #,'2016_TEACHER_STUDENT_RATIO'
        #,'2016_TEACHER_BEGINNING_SALARY_AVERAGE'
        #,'2016_TEACHER_TOTAL_SALARY_AVERAGE'
        #,'2015_INSTRUCTIONAL_EXPENDITURES_RATIO'
        ,'2016_WEALTH_PER_WADA'
        ,'2016_ENROLLMENT'
        #,'2016_WADA'
                ]]
    X = sm.add_constant(X)
    
    est = sm.OLS(y,X).fit()
    print(est.summary())

def createDummies(district, fieldName):
    for value in district[fieldName].unique():
        newFieldName = fieldName+'_'+value.replace(' ','_').upper()
        district[newFieldName] = np.where((district[fieldName]==value),1,0)
    return district

def readDistrictFiles(): 

    #read in csv files
    district = pd.read_csv('tea directory districts.csv', usecols= {0,1,2,3,4}, index_col = 0)
    districtGeometry = pd.read_csv('district_geometry.csv', usecols={'DISTRICT_N','Area'}, index_col = 'DISTRICT_N')
    districtType = pd.read_csv('district type.csv', usecols = {1,3}, index_col = 'District')
    districtRatings = pd.read_csv('district ratings.csv', index_col = 'District_Number', na_values=[' '])  
    districtAEA = pd.read_csv('aea districts.csv', usecols = {1}, index_col = 'District Number')

    

    #read in xlsx files
    districtDistinctions = pd.read_excel('district_distinctions.xlsx', sheetname=0, index_col = 'DISTRICT')
    districtReference = pd.read_excel('district_reference.xlsx', sheetname=0, index_col = 'DISTRICT', na_values=['.'])
    districtStaffStudents = pd.read_excel('TAPR DISTPROF.xlsx', sheetname=0, index_col = 'DISTRICT',
                                          na_values = ['.',-1,-2,-3],
                                          parse_cols = [0,47,89,90,91,92,93,94,95,96,97,99,100,
                                                        101,102,104,123,132,150,171,172,173,247,248,253,257])
    districtECHS = pd.read_excel('Early College High Schools 15-16 list.xlsx', sheetname=0)
    districtChapter41 = pd.read_excel('2015-2016 Revised Final Chapter 41 Districts.xlsx',sheetname=0, parse_cols = [0], index_col = [0])
    districtChapter41Recapture = pd.read_excel('ch41_2016_recapture_funds.xlsx',sheetname=0, parse_cols = [1,3], index_col = [0])
    districtExpenditures = pd.read_excel('Total Operating Expenditures 2015 by School District.xlsx',sheetname=0, index_col =[0])

    #read in xls files
    districtWealth = pd.read_excel('Wealth per WADA 2016.xls', sheetname=0, index_col = 'DISTRICT')

    #get unique values from campus ECHS list as new object
    districtECHS = pd.DataFrame({'DISTRICT_N':districtECHS.DistrictNumber.unique(), 'ECHS_FLAG':'Y'}) 

    #add missing index
    districtECHS.set_index('DISTRICT_N', inplace = True)

    #add flags
    districtReference['MAGNET_DISTRICT_FLAG'] = np.where(districtReference.index==31916,'Y','N')
    districtReference['MILITARY_BASE_FLAG'] = np.where((districtReference.index==15914) | (districtReference.index==15913) | 
                (districtReference.index==15906),'Y','N')
    districtAEA['AEA_FLAG'] = 'Y'
    districtDistinctions['DISTINCTION_FLAG'] = np.where(districtDistinctions['DAD_POST']=='1','Y','N')
    districtChapter41['CHAPTER41_FLAG'] = 'Y'
    districtChapter41Recapture['RECAPTURE_FLAG'] = np.where(districtChapter41Recapture['2016   Total Recapture']>0,'Y','N')

    #remove extra columns
    districtRatings.drop(districtRatings.columns[[0,1,2,3]], axis=1, inplace=True)
    districtReference = districtReference[['DFLCHART','MAGNET_DISTRICT_FLAG','MILITARY_BASE_FLAG'
                                           ,'DI1_MET','DI1','DI2_MET','DI2','DI3_MET',
                                           'DI3','DI4_MET','DI4'
                                           ]]
    districtDistinctions = districtDistinctions[['DISTINCTION_FLAG']]
    districtWealth.drop(districtWealth.columns[[0]], axis=1, inplace=True)

    #rename index columns to standard name
    district.index.names = ['DISTRICT_N']
    districtGeometry.index.names = ['DISTRICT_N']
    districtType.index.names = ['DISTRICT_N']
    districtRatings.index.names = ['DISTRICT_N']
    districtDistinctions.index.names = ['DISTRICT_N']
    districtReference.index.names = ['DISTRICT_N']
    districtStaffStudents.index.names = ['DISTRICT_N']
    districtWealth.index.names = ['DISTRICT_N']
    districtChapter41.index.names = ['DISTRICT_N'] 
    districtChapter41Recapture.index.names = ['DISTRICT_N']
    districtExpenditures.index.names = ['DISTRICT_N']

    #rename other columns
    districtType = districtType.rename(columns = {'Description':'Type_Description'})
    districtReference = districtReference.rename(columns = {'DFLCHART':'CHARTER_OPERATOR'})
    districtStaffStudents = districtStaffStudents.rename(columns =
                                                         {'DPETALLC':'2016_ENROLLMENT','DPEMALLP':'2015_MOBILITY_PCT',
                                                          'DPETWHIP':'2016_WHITE_PCT','DPETBLAP':'2016_AFRAM_PCT',
                                                          'DPETHISP':'2016_HISP_PCT','DPETINDP':'2016_AMIND_PCT',
                                                          'DPETASIP':'2016_ASIAN_PCT','DPETPCIP':'2016_PACIF_PCT',
                                                          'DPETTWOP':'2016_TWO_OR_MORE_PCT',
                                                          'DPETECOP':'2016_ECODIS_PCT','DPETRSKP':'2016_AT_RISK_PCT',
                                                          'DPETLEPP':'2016_LEP_PCT','DPETSPEP':'2016_SPED_PCT',
                                                          'DPETVOCP':'2016_CTE_PCT','DPETGIFP':'2016_GT_PCT',
                                                          'DPSTTOFC':'2016_TEACHER_TOTAL_FTE',
                                                          'DPST00FC':'2016_TEACHER_BEGINNING_FTE',
                                                          'DPSATOFC':'2016_ALL_STAFF_FTE',
                                                          'DPSTURNR':'2016_TEACHER_TURNOVER_RATIO',
                                                          'DPSTTENA':'2016_TEACHER_TENURE_AVERAGE',
                                                          'DPSTEXPA':'2016_TEACHER_EXPERIENCE_AVERAGE',
                                                          'DPSTKIDR':'2016_TEACHER_STUDENT_RATIO',
                                                          'DPFEIERP':'2015_INSTRUCTIONAL_EXPENDITURES_RATIO',
                                                          'DPST00SA':'2016_TEACHER_BEGINNING_SALARY_AVERAGE',
                                                          'DPSTTOSA':'2016_TEACHER_TOTAL_SALARY_AVERAGE'})
    districtWealth = districtWealth.rename(columns = 
                                                        {'School Year 2015-2016 WADA':'2016_WADA',
                                                         'Tax Year 2014 Property Values':'2014_PROP_VAL',
                                                         'School Year 2015-2016 Wealth per WADA':'2016_WEALTH_PER_WADA'})
    districtChapter41Recapture = districtChapter41Recapture.rename(columns = {'2016   Total Recapture':'2016_TOTAL_RECAPTURE'}) 

    return district, districtGeometry, districtType, districtRatings, districtAEA, districtDistinctions, \
        districtReference, districtECHS, districtStaffStudents, districtWealth, districtChapter41, districtChapter41Recapture, \
        districtExpenditures

    

def combineDistrictFiles(district, districtGeometry, districtType, districtRatings, districtAEA,
                                    districtDistinctions, districtReference, districtECHS, districtStaffStudents, 
                                    districtWealth, districtChapter41, districtChapter41Recapture, districtExpenditures):
    district = district.join(districtReference,how='left')
    district = district.join(districtGeometry,how='left')
    district = district.join(districtType,how='left')
    district = district.join(districtAEA,how='left')
    district = district.join(districtECHS,how='left')
    district = district.join(districtRatings,how='inner')
    district = district.join(districtDistinctions,how='left')
    district = district.join(districtStaffStudents,how='left')
    district = district.join(districtWealth,how='left')
    district = district.join(districtChapter41,how='left')
    district = district.join(districtChapter41Recapture, how='left')
    district = district.join(districtExpenditures, how='left')
    
    #convert all column names to upper case for cleanliness
    district.columns = district.columns.str.upper()
    return district



def convertToCategorical(value):

    #Use this function to convert a field to categorical
    #Still need to apply this to the fields somewhow

    if value == "A":
        return 1
    elif value == "B":
        return 2
    elif value == "C":
        return 3
    elif value == "D":
        return 4
    elif value == "F":
        return 5
    else:
        return 6


    

#def convertPredictors(district):
    #use this function to convert the A-F fields to categorical fields with labels
    #consider as alternative
#    grade_cols = [col for col in district.columns if 'GRADE' in col]
#    for i in ob_cols:
#        district[i] = district[i].fillna('Missing')
 #   print(grade_cols)
 #   return district

        

def removeNotRated(district):
    #Remove districts that did not receive any ratings
    grade_cols = [col for col in district.columns if 'GRADE' in col]
    district.dropna(subset = [grade_cols],how='all', inplace=True)
    return district



def fixString(district):
    #change numeric fields to strings (where the number is just an unordered category)
    district['COUNTY_NUMBER'] = district['COUNTY_NUMBER'].apply(str)
    district['REGION_NUMBER'] = district['REGION_NUMBER'].apply(str)
    return district

    

def fixMissing(district):
    #fix flags
    #get flag columns into list
    flag_cols = [col for col in district.columns if 'FLAG' in col]
    for i in flag_cols:
        district[i] = district[i].fillna('N')

    #fix other string cols
    ob_cols = list(district.select_dtypes(include=['object']).columns)
    for i in ob_cols:
        district[i] = district[i].fillna('Missing')
    return district

def dropRowsMissingScores(district,fieldName):
    district=district[np.isfinite(district[fieldName])]
    return district

def toCSV(district):
        district.to_csv('district_combined.csv',sep=',')

    

def histogramPredictors(district):
    #http://seaborn.pydata.org/tutorial/categorical.html
    orderList = district.DOMAIN_I_LETTER_GRADE.unique()
    sns.countplot(x='DOMAIN_I_LETTER_GRADE', data=district, palette="Greens_d", order=sorted(orderList.tolist()));



def logTransform(district):
    district['log2015_TOTAL_OPERATING_EXPENDITURES'] = np.log(district['2015_TOTAL_OPERATING_EXPENDITURES'])
    district['log2015_OPERATING_EXPENDITURES_PER_STUDENT'] = np.log(district['2015_OPERATING_EXPENDITURES_PER_STUDENT'])
    district['logAREA'] = np.log(district['AREA'])
    district['log2016_ENROLLMENT'] = np.log(district['2016_ENROLLMENT'])
    district['log2016_WADA'] = np.log(district['2016_WADA'])
    district['log2014_PROP_VAL'] = np.log(district['2014_PROP_VAL']) 
    district['log2016_WEALTH_PER_WADA'] = np.log(district['2016_WEALTH_PER_WADA']) 
    return district

main()
