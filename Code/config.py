column_list = dict(
    select_columns = ['CLTV',
                    'DDLPI',
                    'DTI',
                    'LTV',
                    # 'PPM_N',
                    'UPB',
                    'actual_loss',
                    'channel_B',
                    'channel_C',
                    'channel_R',
                    'channel_T', #
                    'channel_9', #
                    'credit_score',
                    'current_UPB',
                    'current_deferred_UPB',
                    'current_interest_rate',
                    'deferred_payment_mod_ ',
                    'deferred_payment_mod_N',
                    'deferred_payment_mod_Y',
                    'expenses',
                    'first_time_9',
                    'first_time_N',
                    'first_time_Y',
                    'legal_cost',
                    'loan_age',
                    'maintainance_cost',
                    'mi_recovery',
                    'misc_cost',
                    'modification_cost',
                    'modification_flag_Y',
                    'monthly_period',
                    'mortgage_insurance',
                    'msa',
                    'non_mi_recovery',
                    'num_borrowers',
                    'num_units',
                    'occupancy_I',
                    'occupancy_P',
                    'occupancy_S',
                    'occupancy_9', #
                    'original_interest_rate',
                    # 'original_loan_term',
                    # 'PPM_Y', #
                    # 'PPM_N', #
                    'product_type_FRM',
                    'property_type_CO',
                    'property_type_CP',
                    'property_type_MH',
                    'property_type_PU',
                    'property_type_SF',
                    'purpose_C',
                    'purpose_N',
                    'purpose_P',
                    'remaining_month',
                    'repurchase_flag_N',
                    'repurchase_flag_Y',
                    'state_AK',
                    'state_AL',
                    'state_AR',
                    'state_AZ',
                    'state_CA',
                    'state_CO',
                    'state_CT',
                    'state_DC',
                    'state_DE',
                    'state_FL',
                    'state_GA',
                    'state_GU',
                    'state_HI',
                    'state_IA',
                    'state_ID',
                    'state_IL',
                    'state_IN',
                    'state_KS',
                    'state_KY',
                    'state_LA',
                    'state_MA',
                    'state_MD',
                    'state_ME',
                    'state_MI',
                    'state_MN',
                    'state_MO',
                    'state_MS',
                    'state_MT',
                    'state_NC',
                    'state_ND',
                    'state_NE',
                    'state_NH',
                    'state_NJ',
                    'state_NM',
                    'state_NV',
                    'state_NY',
                    'state_OH',
                    'state_OK',
                    'state_OR',
                    'state_PA',
                    'state_PR',
                    'state_RI',
                    'state_SC',
                    'state_SD',
                    'state_TN',
                    'state_TX',
                    'state_UT',
                    'state_VA',
                    'state_VI',
                    'state_VT',
                    'state_WA',
                    'state_WI',
                    'state_WV',
                    'state_WY',
                    'step_modification_flag_N', #
                    'step_modification_flag_Y', #
                    'super_conforming_flag_N', #
                    'super_conforming_flag_Y', # 
                    'tax'],
                    # 'zero_balance',
                    # 'zero_balance_date'],
    delete_columns = ['first_payment_date', 
                   'mat_date', 
                   'state', 
                   'postal_code', 
                   'loan_number', 
                   'current_status', 
                   'net_sales_procedees'],
)

unique_values = dict(
    unique_first_time = ['Y', 'N', '9'],
    unique_occupancy = ['P', 'I', 'S', '9'],
    unique_channel = ['R', 'B', 'C', 'T', '9'],
    unique_PPM = ['Y', 'N'],
    unique_step_modification_flags = ['N', 'Y'],
    unique_super_conforming_flags = ['N', 'Y'],
    current_status = [
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        '23',
        '24',
        '25',
        '26',
        '27',
        '28',
        '29',
        '30',
        '31',
        '32',
        '33',
        '34',
        '35',
        '36',
        '37',
        '38',
        '39',
        '40',
        '41',
        '45',
        '46', #Prepayment
        'R',
        'XX'
    ],
    unique_states = [
        'AK',
        'AL',
        'AR',
        'AZ',
        'CA',
        'CO',
        'CT',
        'DC',
        'DE',
        'FL',
        'GA',
        'GU',
        'HI',
        'IA',
        'ID',
        'IL',
        'IN',
        'KS',
        'KY',
        'LA',
        'MA',
        'MD',
        'ME',
        'MI',
        'MN',
        'MO',
        'MS',
        'MT',
        'NC',
        'ND',
        'NE',
        'NH',
        'NJ',
        'NM',
        'NV',
        'NY',
        'OH',
        'OK',
        'OR',
        'PA',
        'PR',
        'RI',
        'SC',
        'SD',
        'TN',
        'TX',
        'UT',
        'VA',
        'VI',
        'VT',
        'WA',
        'WI',
        'WV',
        'WY'
    ]
)