import csv
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


def months_between(date1, date2):
    if date1 > date2:
        date1, date2 = date2, date1

    m1 = date1.year * 12 + date1.month
    m2 = date2.year * 12 + date2.month
    months = m2 - m1

    if date1.day > date2.day:
        months -= 1
    elif date1.day == date2.day:
        seconds1 = date1.hour * 3600 + date1.minute + date1.second
        seconds2 = date2.hour * 3600 + date2.minute + date2.second

        if seconds1 > seconds2:
            months -= 1

    return months


_campaign_list = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]

_campaign_dates = {
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


def _get_prev_campaign_ids(curr_campaign_id):
    return list(range(curr_campaign_id + 1, 25))


def campaign_date_to_cal_date(campaign_date):
    return datetime(year=1900 + int(campaign_date // 100),
                    month=int(campaign_date % 100),
                    day=1)


def campaign_date_to_epoch(campaign_date):
    return to_milliseconds(campaign_date_to_cal_date(campaign_date) - EPOCH_TIME)


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

    with open(input_csv_file, 'r') as input_csv:
        csv_reader = csv.DictReader(input_csv)

        # read and process each row
        row_num = 0
        for row in csv_reader:
            age = to_float(row['AGE'])
            income = to_float(row['INCOME'])

            # extract the last 16 campaign data for training
            campaign_states = []

            for campaign_id in _campaign_list:
                campaign_date = row['ADATE_' + str(campaign_id)]
                campaign_date = _campaign_dates.get(campaign_id) if campaign_date is None else campaign_date
                campaign_cal_date = campaign_date_to_cal_date(campaign_date)
                campaign_timestamp = campaign_date_to_epoch(campaign_date)

                prev_campaign_ids = _get_prev_campaign_ids(campaign_id)

                ngiftall = sum([row['RDATE_' + str(x)] is not None and row['RDATE_' + str(x)] > 0
                                for x in prev_campaign_ids])
                numprom = sum([row['ADATE_' + str(x)] is not None and row['ADATE_' + str(x)] > 0
                               for x in prev_campaign_ids])
                frequency = float(ngiftall) / numprom

                recency = 0
                lastgift = 0
                last_gift_cal_date = None
                if ngiftall > 0:
                    for prev_campaign_id in prev_campaign_ids:
                        if row['RDATE_' + str(prev_campaign_id)] is not None and row['RDATE_' + str(prev_campaign_id)] > 0:
                            last_id = id
                            break

                    last_gift_date = row['RDATE_' + str(last_id)]
                    last_gift_cal_date = campaign_date_to_cal_date(last_gift_date)

                    recency = months_between(campaign_cal_date, last_gift_cal_date)
                    lastgift = to_float(row['RAMNT_' + str(last_id)])

                ramntall = sum([to_float(row['RAMNT_' + str(x)])
                                    for x in prev_campaign_ids if row['RAMNT_' + str(x)] is not None and
                                                                    row['RAMNT_' + str(x)] > 0])

                nrecproms = 0
                nrecgifts = 0
                totrecamt = 0.0
                recamtpergift = 0.0
                for prev_campaign_id in prev_campaign_ids:
                    prev_prom_date = row['ADATE_' + str(prev_campaign_id)]
                    if prev_prom_date is not None and prev_prom_date > 0:
                        prev_prom_cal_date = campaign_date_to_cal_date(prev_prom_date)

                        if months_between(campaign_cal_date, prev_prom_cal_date) <= 6:
                            nrecproms += 1

                    prev_gift_date = row['RDATE_' + str(prev_campaign_id)]
                    if prev_gift_date is not None and prev_gift_date > 0:
                        prev_gift_cal_date = campaign_date_to_cal_date(prev_gift_date)

                        prev_gift = to_float(row['RAMNT_' + str(prev_campaign_id)])

                        if months_between(campaign_cal_date, prev_gift_cal_date) <= 6:
                            nrecgifts += 1
                            totrecamt += prev_gift

                if totrecamt > 0 and nrecgifts > 0:
                    recamtpergift = float(totrecamt) / nrecgifts

                last_prom_cal_date = None
                for prev_campaign_id in prev_campaign_ids:
                    prev_prom_date = row['ADATE_' + str(prev_campaign_id)]
                    if prev_prom_date is not None and prev_prom_date > 0:
                        last_prom_cal_date = campaign_date_to_cal_date(prev_prom_date)
                        break

                promrecency = 0 if last_prom_cal_date is None else months_between(campaign_cal_date, last_prom_cal_date)

                first_prom_cal_date = None
                for first_campaign_id in reversed(prev_campaign_ids):
                    first_prom_date = row['ADATE_' + str(first_campaign_id)]
                    if first_prom_date is not None and first_prom_date > 0:
                        first_prom_cal_date = campaign_date_to_cal_date(first_prom_date)
                        break

                first_gift_cal_date = None
                for first_campaign_id in reversed(prev_campaign_ids):
                    first_gift_date = row['RDATE_' + str(first_campaign_id)]
                    if first_gift_date is not None and first_gift_date > 0:
                        first_gift_cal_date = campaign_date_to_cal_date(first_gift_date)
                        break

                timelag = 0
                if first_prom_cal_date is not None and first_gift_cal_date is not None:
                    timelag = months_between(first_gift_cal_date, first_prom_cal_date)

                recencyratio = float(recency) / timelag if timelag > 0 else 0.0
                promrecratio = float(promrecency) / timelag if timelag > 0 else 0.0

                respondedbit1 = 0
                respondedbit2 = 0
                respondedbit3 = 0
                for prev_campaign_id in prev_campaign_ids:
                    prev_gift_date = row['RDATE_' + str(prev_campaign_id)]
                    if prev_gift_date is not None and prev_gift_date > 0:
                        prev_gift_cal_date = campaign_date_to_cal_date(prev_gift_date)
                        months = months_between(campaign_cal_date, prev_gift_cal_date)

                        if months == 1:
                            respondedbit1 = 1
                        elif months == 2:
                            respondedbit2 = 1
                        elif months == 3:
                            respondedbit3 = 1

                mailedbit1 = 0
                mailedbit2 = 0
                mailedbit3 = 0
                for prev_campaign_id in prev_campaign_ids:
                    prev_prom_date = row['ADATE_' + str(prev_campaign_id)]
                    if prev_prom_date is not None and prev_prom_date > 0:
                        prev_prom_cal_date = campaign_date_to_cal_date(prev_prom_date)
                        months = months_between(campaign_cal_date, prev_prom_cal_date)

                        if months == 1:
                            mailedbit1 = 1
                        elif months == 2:
                            mailedbit2 = 1
                        elif months == 3:
                            mailedbit3 = 1

                campaign_state = {
                    'id':             row_num,
                    'age':            age,
                    'income':         income,
                    'ngiftall':       ngiftall,
                    'numprom':        numprom,
                    'frequency':      frequency,
                    'recency':        recency,
                    'lastgift':       lastgift,
                    'ramntall':       ramntall,
                    'nrecproms':      nrecproms,
                    'nrecgifts':      nrecgifts,
                    'totrecamt':      totrecamt,
                    'recamtpergift':  recamtpergift,
                    'promrecency':    promrecency,
                    'timelag':        timelag,
                    'recencyratio':   recencyratio,
                    'promrecratio':   promrecratio,
                    'respondedbit1':  respondedbit1,
                    'respondedbit2':  respondedbit2,
                    'respondedbit3':  respondedbit3,
                    'mailedbit1':     mailedbit1,
                    'mailedbit2':     mailedbit2,
                    'mailedbit3':     mailedbit1
                }

                campaign_states.append(campaign_state)

            # write out the training data file in (state, action, reward, next_state)
            

            row_num += 1


def main(args):
    pass


if __name__ == '__main__':
    main(sys.argv)

    sys.exit(0)
