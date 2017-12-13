import argparse
import csv
import os
import sys

from datetime import datetime

DEFAULT_MAILING_COST = 0.68

DEFAULT_NONE_VALUE = 0.0

EPOCH_TIME = datetime.utcfromtimestamp(0)


def to_float(val):
    try:
        float_val = float(val)
    except ValueError:
        float_val = DEFAULT_NONE_VALUE

    return float_val


def to_milliseconds(time_delta):
    return time_delta.days * 86400000 + time_delta.seconds * 1000 + time_delta.microseconds / 1000


def campaign_date_to_epoch(campaign_date):
    curr_campaign_date = datetime(year=1900 + int(campaign_date // 100),
                                  month=int(campaign_date % 100),
                                  day=1)

    return to_milliseconds(curr_campaign_date - EPOCH_TIME)


def process_sample_data(input_csv_file, output_csv_file):
    """
    Extract features from raw data:
    01. age:              individual’s age
    02. income:           income bracket
    03. ngiftall:         number of gifts to date
    04. numprom:          number of promotions to date
    05. frequency:        ngiftall / numprom
    06. recency:          number of months since last gift
    07. lastgift:         amount in dollars of last gift
    08. ramntall:         total amount of gifts to date
    09. nrecproms:        num. of recent promotions(last6 mo.)
    10. nrecgifts:        num. of recent gifts(last6 mo.)
    11. totrecamt:        total amount of recent gifts(6mo.)
    12. recamtpergift:    recent gift amount per gift(6mo.)
    13. promrecency:      num. of months since last promotion
    14. timelag:          num. of mo’s from first prom to gift
    15. recencyratio:     recency / timelag
    16. promrecratio:     promrecency / timelag
    17. respondedbit[1]:  whether responded last month
    18. respondedbit[2]:  whether responded 2 months ago
    19. respondedbit[3]:  whether responded 3 months ago
    20. mailedbit[1]:     whether promotion mailed last month
    21. mailedbit[2]:     whether promotion mailed 2 mo’s ago
    22. mailedbit[3]:     whether promotion mailed 3 mo’s ago

    Recorded Action:
    action:               whether mailed in current promotion
    """
    campaign_list = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]

    campaign_dates = {
        3:  9606,
        4:  9604,
        5:  9604,
        6:  9603,
        7:  9602,
        8:  9601,
        9:  9511,
        10: 9510,
        11: 9510,
        12: 9508,
        13: 9507,
        14: 9506,
        15: 9504,
        16: 9503,
        17: 9502,
        18: 9501
    }

    with open(input_csv_file, 'r') as input_csv:
        csv_reader = csv.DictReader(input_csv)

        # read and process each row
        row_num = 0
        for row in csv_reader:
            age = to_float(row['AGE'])
            income = to_float(row['INCOME'])

            # extract the last 16 campaign data for training
            campaign_states = []

            # 1st campaign
            campaign_id = campaign_list[0]
            campaign_date = row['ADATE_' + str(campaign_id)]
            campaign_date = campaign_dates.get(campaign_id) if campaign_date is None else campaign_date
            campaign_timestamp = campaign_date_to_epoch(campaign_date)

            


            campaign_state = {
                'id':       row_num,
                'age':      age,
                'income':   income
            }




            # write out the training data file

            row_num += 1







def main(args):
    pass

if __name__ == '__main__':
    main(sys.argv)

    sys.exit(0)
