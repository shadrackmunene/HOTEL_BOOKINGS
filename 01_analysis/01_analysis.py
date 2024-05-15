# LIBRARIES ----
# Data Analysis:
import pandas as pd
import numpy as np
import pytimetk as tk
from missingno import matrix

# Modelling:
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit
from sklearn.metrics import roc_auc_score

# File management:
import os
os.getcwd()
"""path="c:\\Users\\smunene\\Desktop\\Private\\Classwork\\Matt\\hotel_bookings"
os.chdir(path)"""
os.getcwd()

# DATA----
H1 = pd.read_csv("00_data/H1.csv")\
    .assign(hotel= "Resort Hotels")
H2 = pd.read_csv("00_data/H2.csv")\
    .assign(hotel= lambda x: "Restuarants")

hotel_cancellations_raw_df = pd.concat([H1, H2], axis=0)
columns=hotel_cancellations_raw_df.columns
columns=['IsCanceled', 'LeadTime', 'ArrivalDateYear', 'ArrivalDateMonth',
       'ArrivalDateWeekNumber', 'ArrivalDateDayOfMonth',
       'StaysInWeekendNights', 'StaysInWeekNights', 'Adults', 'Children',
       'Babies', 'Meal', 'Country', 'MarketSegment', 'DistributionChannel',
       'IsRepeatedGuest', 'PreviousCancellations',
       'PreviousBookingsNotCanceled', 'ReservedRoomType', 'AssignedRoomType',
       'BookingChanges', 'DepositType', 'Agent', 'Company',
       'DaysInWaitingList', 'CustomerType', 'ADR', 'RequiredCarParkingSpaces',
       'TotalOfSpecialRequests', 'ReservationStatus', 'ReservationStatusDate',
       'hotel']
# To transform the column names:
import re
transformed_columns = [
    re.sub(r'([A-Z])', r'_\1', col).lower().lstrip('_') for col in columns
]

#columns_with_underscores 
transformed_columns
# Applying this transformation to a DataFrame
hotel_cancellations_raw_df.columns = transformed_columns
hotel_cancellations_raw_df
hotel_cancellations_raw_df.to_csv("00_data/hotel_cancellations_raw_df.csv")

pd.read_csv("00_data/hotel_cancellations_raw_df.csv")

# COST ANALYSIS ----

# How many people cancel?
cancellation_counts = hotel_cancellations_raw_df['is_canceled'].value_counts().reset_index()
cancellation_counts.columns = ['is_canceled','n']
cancellation_counts['prop'] = cancellation_counts['n']/cancellation_counts["n"].sum()

# Display the resul
cancellation_counts

# Hoe much does this cost the hotel?
cost =44224*200*3.4
formatted_cost="${:,.0f}".format(cost)
formatted_cost

# 10% of the cost
ten_pct_cost = 0.1*cost
formatted_ten_pct_cost = "${:,.0f}".format(ten_pct_cost)
formatted_ten_pct_cost

# 1.0 DATA PROCESSING ----
hotel_cancellations_df = hotel_cancellations_raw_df.copy()
hotel_cancellations_df

# Add features
hotel_cancellations_df['total_stay'] = hotel_cancellations_df['stays_in_weekend_nights'] + hotel_cancellations_df['stays_in_week_nights']

hotel_cancellations_df['total_guests'] = hotel_cancellations_df['adults'] + hotel_cancellations_df['children'] + hotel_cancellations_df['babies']

hotel_cancellations_df['diffrent_room_assigned'] = (hotel_cancellations_df['reserved_room_type'] != hotel_cancellations_df['assigned_room_type']).astype(int) 

# Dropping the orriginal columns
hotel_cancellations_df.drop(
    columns=[
        'stays_in_weekend_nights',
        'stays_in_week_nights',
        'adults',
        'children',
        'babies',
        'reserved_room_type',
        'assigned_room_type'
    ],
    inplace=True
)

hotel_cancellations_df.glimpse()

# Fix Missing data --------------------------------
# Visualize missing data
matrix(hotel_cancellations_df)

# Count by Country with sorting
country_count = hotel_cancellations_df['country']\
    .value_counts()\
    .reset_index()
country_count.columns=['country', 'count']
country_count

# Fix missing values nad drop columns
hotel_cancellations_df = hotel_cancellations_df.drop(
    columns= ['company','agent']
    hotel_cancellations_df['country'] =hotel_cancellations_df['country'].fillna('PRT')
)
hotel_cancellations_df.dropna(inplace=True)

# dROP Unused columns --------------------------------

hotel_cancellations_df.glimpse()
hotel_cancellations_df.drop(
    columns=['reservation_status_date'],inplace=True
)

# Initial complete Proecessing--------------------------------
hotel_cancellations_df.glimpse()

#hotel_cancellations_df.to_csv('00_data/hotel_cancellations_clean.csv', index=False)

# 2.0 CORRELATION ANALYSIS (1: Association)----------------------------------------------------------------
hotel_cancellations_df= pd.read_csv('00_data/hotel_cancellations_clean.csv')

# Binarize------------------------

df_binarized=hotel_cancellations_df.binarize()
df_binarized.glimpse()



T





