import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


FILES = ['ss13husa.csv', 'ss13husb.csv']
OUTPUT = 'households.p'

def get_data(files):
    
    """
    Reads data from csv flat files
    
    Args:
        FILES (list): List of csv flat files
    Returns:
        df (pandas.DataFrame): Dataframe of data
    """

    for i, f in enumerate(files):
        if i == 0:
            df = pd.read_csv(f)
        else:
            df = df.append(pd.read_csv(f))
    return df

def create_internet_column(df):

    """
    Appends an internet column to the dataframe. Internet is defined as true if a 
    household has DSL, DIALUP, SATELLITE, OTHSVCEX, MODEM, or FIBEROP.
    
    Args:
        df (pandas.DataFrame): Dataframe of data w/o internet column
    Returns:
        df (pandas.DataFrame): Dataframe of data w/ internet column
    """

    # Define Internet Features
    internet_features = ['DSL','DIALUP', 'SATELLITE', 'OTHSVCEX', 'MODEM', 'FIBEROP']
    int_true = {}
    int_false = {}
    # Condition which represents having some form of internet service
    for feature in internet_features:
        int_true[feature] = df[feature] == 1
    
    # Obtain indexes for rows in dataframe that have some form of internet service
    for i, df_temp in enumerate(int_true.values()):
        if i == 0:
            internet_true = df_temp
        else:
            internet_true = internet_true | df_temp    
    
    # Create INTERNET column in dataframe based on indexes of internet_true
    df['INTERNET'] = 1
    df['INTERNET'] = df['INTERNET'].where(internet_true, other = 0)

    return df

def drop_irrelevant_features(df):

    """
    Performs feature selection to only keep the relevant features.
    
    Args:
        df (pandas.DataFrame): Dataframe of data w/ all features
    Returns:
        df (pandas.DataFrame): Dataframe of data w/ selected features
    """

    features = [#'RT',          # Record Type
     #'SERIALNO',    # Housing Unit Serial No.
     #'DIVISION',    # Division Code
     #'PUMA',        # Public Use Microdata Area Code
     'REGION',      # Region
     'ST',          # State Code
     #'ADJHSG',     # Adjustment factor for Housing
     #'ADJINC',     # Inflation adjustment for Income
     #'WGTP',       # Housing Weight
     'NP',          # Number of person records following this household
     'TYPE',        # Type of Unit 
     #'ACR',         # Lot Size
     #'AGS',        # Sales of Agriculture Products
     'BATH',        # Bathtub or Shower
     'BDSP',        # Number of bedrooms
     'BLD',         # Units in Structure
     'BROADBND',    # Mobile Broadband Plan
     #'BUS',         # Business or medical office on property
     #'COMPOTHX',   # Other computer equipment
     #'CONP',       # Condo Fee
     #'ELEP',       # Electricity Monthly Cost
     'FS',          # Yearly Food Stamp Recipiency
     #'FULP',       # Fuel Cost (other than gas and elec)
     #'GASP',       # Monthly Gas Cost
     'HANDHELD',    # Handheld Computer
     #'HFL',        # House Heating Fuel
     #'INSP',       # Fire/Hazard/Flood Insurance
     'LAPTOP',      # Laptop?
     #'MHP',        # Mobile Home Costs
     #'MRGI',       # First Mortgage includes fire/hazard/flood insurance
     #'MRGP',       # First Mortgage Payment
     #'MRGT',       # First Mortgage Payment includes real estate taxes
     #'MRGX',       # First Mortgage Status
     #'REFR',       # Refridgerator
     'RMSP',        # Number of Rooms
     #'RNTM',        # Meals included in rent
     #'RNTP',        # Monthly Rent
     #'RWAT',       # Hot and Cold Water Running
     #'RWATPR',     # Running Water
     'SINK',        # Sink with a Faucet
     #'SMP',        # Total payment on all second and junior mortgages
     'STOV',        # Stove or Range
     'TEL',         # Telephone
     'TEN',         # Tenure
     #'TOIL',       # Flush Toilet
     #'VACS',        # Vacancy Status
     #'VALP',        # Property Value
     #'VEH',        # Vehicles Available
     #'WATP',       # Water Yearly Cost
     'YBL',         # When Structure first built
     #'FES',         # Family Type and Employment Status
     #'FFINCP',      # Family income allocation flag      
     #'FGRNTP',      # Yearly food stamp allocation flag
     #'FHINCP',      # Household income allocation flag
     #'FINCP',       # Family Income
     #'FPARC',       # Family presence and age of related children
     #'FSMOCP',      # Selected monthly owner cost allocation flag
     #'GRNTP',       # Gross Rent
     #'GRPIP',       # Gross rent as a percentage of household income
     'HHL',         # Household Language
     'HHT',         # Household/family Type
     'HINCP',       # Household income (past 12 months)
     'HUGCL',       # Household w/ grandparent living w/ grandchildren
     'HUPAC',       # Children Present
     #'HUPAOC',     # HH presence and age of own children
     #'HUPARC',     # HH presence and age of related children
     'KIT',         # Complete kitchen facilities
     'LNGI',        # Limited English Speaking Households
     'MULTG',       # Multigenerational Household
     #'MV',          # When moved into this house or apartment
     'NOC',         # Number of children in household
     #'NPF',         # Number of persons in family
     'NPP',         # Grandparent headed household w/ no parent present
     'NR',          # Presence of nonrelative in household
     'NRC',         # Number of related children in household
     #'OCPIP',       # Selected monthly owner costs as a percentage of household income
     'PARTNER',     # Unmarried partner household
     'PLM',         # Complete plumbing facilities
     'PSF',         # Presence of subfamilies in Household
     'R18',         # Presence of persons under 18 years in household
     'R60',         # Presence of persons 60 years and over in household
     'R65',         # Presence of persons 65 years and over in household
     #'RESMODE',    # Response Mode
     #'SMOCP',      # Selected Monthly Owner Costs
     #'SMX',        # Second or junior mortgage home equity loan status
     #'SRNT',       # Specified Rent Unit
     'SSMC',        # Same Sex Marriage Households
     'SVAL',        # Specifice value owner unit
     #'TAXP',        # Property Taxes (yearly)
     #'WIF',         # Workers in family during past 12 months
     #'WKEXREL'     # Work experience of householder or spouse
     #'WORKSTAT'    # Work status of householder or spouse in family households
     #'ACCESS',      # Access to the Internet
     #'DIALUP',      # Dial-up Service
     #'DSL',         # DSL Service
     #'FIBEROP',     # Fiber-optic Internet Service
     #'MODEM',       # Cable Internet Service
     #'OTHSVCEX',   # Other internet services
     #'SATELLITE',   # Satellite Internet Service
     'INTERNET'     # Flag indicator of internet service
    ]
    return df[features]

def drop_missing_data(df):

    """
    Appends an internet column to the dataframe. Internet is defined as true if a 
    household has DSL, DIALUP, SATELLITE, OTHSVCEX, MODEM, or FIBEROP.
    
    Args:
        df (pandas.DataFrame): Dataframe of data w/ NaNs
    Returns:
        df (pandas.DataFrame): Dataframe of data w/o NaNs
    """

    return df.dropna()

def specify_data_types(df):

    """
    Sets the type for categorical variables as objects and sets the type for continuous
    variables.
    
    Args:
        df (pandas.DataFrame): Dataframe of data
    Returns:
        df (pandas.DataFrame): Dataframe of data
    """

    df['REGION'] = df['REGION'].astype(object)      # Region, obj
    df['ST'] = df['ST'].astype(object)          # State Code, obj
    df['NP'] = df['NP'].astype(int)          # Number of person records following this household, int
    df['TYPE'] = df['TYPE'].astype(object)        # Type of Unit, obj 
    df['BATH'] = df['BATH'].astype(object)        # Bathtub or Shower, obj
    df['BDSP'] = df['BDSP'].astype(int)        # Number of bedrooms, int
    df['BLD'] = df['BLD'].astype(object)         # Units in Structure, obj
    df['BROADBND'] = df['BROADBND'].astype(object)    # Mobile Broadband Plan, obj
    df['FS'] = df['FS'].astype(object)          # Yearly Food Stamp Recipiency, obj
    df['HANDHELD'] = df['HANDHELD'].astype(object)    # Handheld Computer, obj
    df['LAPTOP'] = df['LAPTOP'].astype(object)      # Laptop?, obj
    df['RMSP'] = df['RMSP'].astype(int)        # Number of Rooms, int
    df['SINK'] = df['SINK'].astype(object)        # Sink with a Faucet, obj
    df['STOV'] = df['STOV'].astype(object)        # Stove or Range, obj
    df['TEL'] = df['TEL'].astype(object)         # Telephone, obj
    df['TEN'] = df['TEN'].astype(object)         # Tenure, obj
    df['YBL'] = df['YBL'].astype(object)         # When Structure first built, obj
    df['HHL'] = df['HHL'].astype(object)         # Household Language, obj
    df['HHT'] = df['HHT'].astype(object)         # Household/family Type, obj
    df['HINCP'] = df['HINCP'].astype(float)       # Household income (past 12 months), float
    df['HUGCL'] = df['HUGCL'].astype(object)       # Household w/ grandparent living w/ grandchildren, obj
    df['HUPAC'] = df['HUPAC'].astype(object)       # Children Present, obj
    df['KIT'] = df['KIT'].astype(object)         # Complete kitchen facilities, obj
    df['LNGI'] = df['LNGI'].astype(object)        # Limited English Speaking Households, obj
    df['MULTG'] = df['MULTG'].astype(object)       # Multigenerational Household, obj
    df['NOC'] = df['NOC'].astype(int)         # Number of children in household, int
    df['NPP'] = df['NPP'].astype(object)         # Grandparent headed household w/ no parent present, obj
    df['NR'] = df['NR'].astype(object)          # Presence of nonrelative in household, obj
    df['NRC'] = df['NRC'].astype(int)         # Number of related children in household, int
    df['PARTNER'] = df['PARTNER'].astype(object)     # Unmarried partner household, obj
    df['PLM'] = df['PLM'].astype(object)         # Complete plumbing facilities, obj
    df['PSF'] = df['PSF'].astype(object)         # Presence of subfamilies in Household, obj
    df['R18'] = df['R18'].astype(object)         # Presence of persons under 18 years in household, obj
    df['R60'] = df['R60'].astype(object)         # Presence of persons 60 years and over in household, obj
    df['R65'] = df['R65'].astype(object)         # Presence of persons 65 years and over in household, obj
    df['SSMC'] = df['SSMC'].astype(object)        # Same Sex Marriage Households, obj
    df['SVAL'] = df['SVAL'].astype(object)        # Specifice value owner unit, obj
    df['INTERNET'] = df['INTERNET'].astype(int)      # Flag indicator of internet service, obj
    return df

def main():
    df = get_data(FILES)
    df = create_internet_column(df)
    df = drop_irrelevant_features(df)
    df = drop_missing_data(df)
    df = specify_data_types(df)
    pickle.dump(df, open(OUTPUT, 'wb'))

if __name__ == "__main__":
    main()
