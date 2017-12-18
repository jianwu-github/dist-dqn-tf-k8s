import csv
import sys

from datetime import datetime

DEFAULT_MAILING_COST = 0.68

DEFAULT_NONE_VALUE = 0

EPOCH_TIME = datetime.utcfromtimestamp(0)


def to_float(val):
    try:
        float_val = float(val)
    except ValueError:
        float_val = float(DEFAULT_NONE_VALUE)

    return float_val


def to_milliseconds(time_delta):
    return int(time_delta.days * 86400000 + time_delta.seconds * 1000 + time_delta.microseconds / 1000)


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
    2:  9706,
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
    campaign_date_val = int(campaign_date)
    return datetime(year=1900 + int(campaign_date_val // 100),
                    month=int(campaign_date_val % 100),
                    day=1)


def campaign_date_to_epoch(campaign_date):
    return to_milliseconds(campaign_date_to_cal_date(campaign_date) - EPOCH_TIME)


def process_input_data(input_csv_file, output_csv_file):
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
    13. recamtperprom:    recent gift amount per prom(6mo.)
    14. promrecency:      num. of months since last promotion
    15. timelag:          num. of mo’s from first prom to gift
    16. recencyratio:     recency / timelag
    17. promrecratio:     promrecency / timelag
    18. respondedbit[1]:  whether responded last month
    19. respondedbit[2]:  whether responded 2 months ago
    20. respondedbit[3]:  whether responded 3 months ago
    21. mailedbit[1]:     whether promotion mailed last month
    22. mailedbit[2]:     whether promotion mailed 2 mo’s ago
    23. mailedbit[3]:     whether promotion mailed 3 mo’s ago

    Recorded Action:
    action:               whether mailed in current promotion
    """
    with open(input_csv_file, 'r') as input_csv, open(output_csv_file, 'w') as output_csv:
        # fix "_csv.Error: line contains NULL byte"
        csv_reader = csv.DictReader(x.replace('\0', '') for x in input_csv)

        training_file_headers = ['correlation_id',
                                 'timestamp',
                                 'state',
                                 'action',
                                 'reward',
                                 'next_state',
                                 'done']

        csv_writer = csv.DictWriter(output_csv, fieldnames=training_file_headers)
        csv_writer.writeheader()

        # read and process each row
        row_num = 0
        for row in csv_reader:
            age = to_float(row['AGE'])
            income = to_float(row['INCOME'])

            # extract the last 16 campaign data for training
            campaign_states = []

            for campaign_id in _campaign_list:
                campaign_date = row['ADATE_' + str(campaign_id)]
                if campaign_date is not None and campaign_date.strip() and int(campaign_date) > 0:
                    curr_campaign_date = campaign_date
                else:
                    curr_campaign_date = _campaign_dates.get(campaign_id)

                curr_campaign_cal_date = campaign_date_to_cal_date(curr_campaign_date)
                curr_campaign_timestamp = campaign_date_to_epoch(curr_campaign_date)

                action = DEFAULT_NONE_VALUE
                if campaign_date is not None and campaign_date.strip() and int(campaign_date) > 0:
                    action = 1

                if campaign_id > 2:
                    curr_donation = row['RAMNT_' + str(campaign_id)]
                    reward = float(DEFAULT_NONE_VALUE) if action == 0 else (-1 * DEFAULT_MAILING_COST)

                    # print("curr_donation: " + curr_donation)

                    if curr_donation is not None and curr_donation.strip() and float(curr_donation) > 0:
                        reward = float(curr_donation) + reward
                else:
                    # no donation info for the latest campaign id = 2
                    reward = float(DEFAULT_NONE_VALUE)

                prev_campaign_ids = _get_prev_campaign_ids(campaign_id)

                ngiftall = 0
                for prev_campaign_id in prev_campaign_ids:
                    prev_donation_date = row['RDATE_' + str(prev_campaign_id)]
                    if prev_donation_date is not None and prev_donation_date.strip() and int(prev_donation_date) > 0:
                        ngiftall += 1

                numprom = 0
                for prev_campaign_id in prev_campaign_ids:
                    prev_prom_date = row['ADATE_' + str(prev_campaign_id)]
                    if prev_prom_date is not None and prev_prom_date.strip() and int(prev_prom_date) > 0:
                        numprom += 1

                frequency = float(ngiftall) / numprom if numprom > 0 else float(DEFAULT_NONE_VALUE)

                recency = 0
                lastgift = 0
                if ngiftall > 0:
                    for prev_campaign_id in prev_campaign_ids:
                        prev_prom_date = row['RDATE_' + str(prev_campaign_id)]
                        if prev_prom_date is not None and prev_prom_date.strip() and int(prev_prom_date) > 0:
                            last_id = prev_campaign_id
                            break

                    last_gift_date = row['RDATE_' + str(last_id)]
                    last_gift_cal_date = campaign_date_to_cal_date(last_gift_date)

                    recency = months_between(curr_campaign_cal_date, last_gift_cal_date)
                    lastgift = to_float(row['RAMNT_' + str(last_id)])

                    related_prom_date = row['RDATE_' + str(last_id)]
                    if related_prom_date is not None and related_prom_date.strip() and int(related_prom_date) > 0:
                        lastgift = lastgift - DEFAULT_MAILING_COST

                # including mailing cost when computing reward
                ramntall = float(DEFAULT_NONE_VALUE)
                for prev_campaign_id in prev_campaign_ids:
                    prev_prom_date = row['ADATE_' + str(prev_campaign_id)]
                    prev_donation = row['RAMNT_' + str(prev_campaign_id)]

                    prev_ramnt = float(DEFAULT_NONE_VALUE)
                    if prev_donation is None or not prev_donation.strip():
                        prev_ramnt = 0.0
                    else:
                        prev_ramnt = float(prev_donation)

                        if prev_prom_date is not None and prev_prom_date.strip() and int(prev_prom_date) > 0:
                            prev_ramnt = prev_ramnt - DEFAULT_MAILING_COST

                    ramntall += prev_ramnt

                nrecproms = 0
                nrecgifts = 0
                totrecamt = 0.0
                recamtpergift = 0.0
                recamtperprom = 0.0
                for prev_campaign_id in prev_campaign_ids:
                    prev_prom_date = row['ADATE_' + str(prev_campaign_id)]
                    prev_action = 0
                    if prev_prom_date is not None and prev_prom_date.strip() and int(prev_prom_date) > 0:
                        prev_prom_cal_date = campaign_date_to_cal_date(prev_prom_date)
                        prev_action = 1

                        if months_between(curr_campaign_cal_date, prev_prom_cal_date) <= 6:
                            nrecproms += 1

                    prev_gift_date = row['RDATE_' + str(prev_campaign_id)]
                    if prev_gift_date is not None and prev_gift_date.strip() and int(prev_gift_date) > 0:
                        prev_gift_cal_date = campaign_date_to_cal_date(prev_gift_date)

                        prev_gift = to_float(row['RAMNT_' + str(prev_campaign_id)])
                        if prev_action == 1:
                            prev_gift = prev_gift - DEFAULT_MAILING_COST

                        if months_between(curr_campaign_cal_date, prev_gift_cal_date) <= 6:
                            nrecgifts += 1
                            totrecamt += prev_gift

                if totrecamt > 0 and nrecgifts > 0:
                    recamtpergift = float(totrecamt) / nrecgifts

                if totrecamt > 0 and nrecproms > 0:
                    recamtperprom = float(totrecamt) / nrecproms

                last_prom_cal_date = None
                for prev_campaign_id in prev_campaign_ids:
                    prev_prom_date = row['ADATE_' + str(prev_campaign_id)]
                    if prev_prom_date is not None and prev_prom_date.strip() and int(prev_prom_date) > 0:
                        last_prom_cal_date = campaign_date_to_cal_date(prev_prom_date)
                        break

                promrecency = 0 if last_prom_cal_date is None else months_between(curr_campaign_cal_date, last_prom_cal_date)

                first_prom_cal_date = None
                for first_campaign_id in reversed(prev_campaign_ids):
                    first_prom_date = row['ADATE_' + str(first_campaign_id)]
                    if first_prom_date is not None and first_prom_date.strip() and int(first_prom_date) > 0:
                        first_prom_cal_date = campaign_date_to_cal_date(first_prom_date)
                        break

                first_gift_cal_date = None
                for first_campaign_id in reversed(prev_campaign_ids):
                    first_gift_date = row['RDATE_' + str(first_campaign_id)]
                    if first_gift_date is not None and first_gift_date.strip() and int(first_gift_date) > 0:
                        first_gift_cal_date = campaign_date_to_cal_date(first_gift_date)
                        break

                timelag = 0
                if first_prom_cal_date is not None and first_gift_cal_date is not None:
                    timelag = months_between(first_gift_cal_date, first_prom_cal_date)

                    # plus 1 means less than one month count as one month to produce recencyratio and promrecration
                    timelag += 1

                recencyratio = float(recency) / timelag if timelag > 0 else 0.0
                promrecratio = float(promrecency) / timelag if timelag > 0 else 0.0

                respondedbit1 = 0
                respondedbit2 = 0
                respondedbit3 = 0
                for prev_campaign_id in prev_campaign_ids:
                    prev_gift_date = row['RDATE_' + str(prev_campaign_id)]
                    if prev_gift_date is not None and prev_gift_date.strip() and int(prev_gift_date) > 0:
                        prev_gift_cal_date = campaign_date_to_cal_date(prev_gift_date)
                        months = months_between(curr_campaign_cal_date, prev_gift_cal_date)

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
                    if prev_prom_date is not None and prev_prom_date.strip() and int(prev_prom_date) > 0:
                        prev_prom_cal_date = campaign_date_to_cal_date(prev_prom_date)
                        months = months_between(curr_campaign_cal_date, prev_prom_cal_date)

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
                    'recamtperprom':  recamtperprom,
                    'promrecency':    promrecency,
                    'timelag':        timelag,
                    'recencyratio':   recencyratio,
                    'promrecratio':   promrecratio,
                    'respondedbit1':  respondedbit1,
                    'respondedbit2':  respondedbit2,
                    'respondedbit3':  respondedbit3,
                    'mailedbit1':     mailedbit1,
                    'mailedbit2':     mailedbit2,
                    'mailedbit3':     mailedbit3,
                    'timestamp':      curr_campaign_timestamp,
                    'action':         action,
                    'reward':         reward
                }

                campaign_states.append(campaign_state)

            # write out the training data file in (state, action, reward, next_state)
            training_data = {}
            for i in range(16):
                curr_state = campaign_states[i]
                next_state = campaign_states[i + 1]

                training_data['correlation_id'] = str(curr_state['id'])
                training_data['timestamp'] = str(curr_state['timestamp'])
                training_data['state'] = str([curr_state['age'],
                                              curr_state['income'],
                                              curr_state['ngiftall'],
                                              curr_state['numprom'],
                                              curr_state['frequency'],
                                              curr_state['recency'],
                                              curr_state['lastgift'],
                                              curr_state['ramntall'],
                                              curr_state['nrecproms'],
                                              curr_state['nrecgifts'],
                                              curr_state['totrecamt'],
                                              curr_state['recamtpergift'],
                                              curr_state['recamtperprom'],
                                              curr_state['promrecency'],
                                              curr_state['timelag'],
                                              curr_state['recencyratio'],
                                              curr_state['promrecratio'],
                                              curr_state['respondedbit1'],
                                              curr_state['respondedbit2'],
                                              curr_state['respondedbit3'],
                                              curr_state['mailedbit1'],
                                              curr_state['mailedbit2'],
                                              curr_state['mailedbit3']
                                              ])
                training_data['action'] = str(curr_state['action'])
                training_data['reward'] = str(curr_state['reward'])
                training_data['next_state'] = str([next_state['age'],
                                                   next_state['income'],
                                                   next_state['ngiftall'],
                                                   next_state['numprom'],
                                                   next_state['frequency'],
                                                   next_state['recency'],
                                                   next_state['lastgift'],
                                                   next_state['ramntall'],
                                                   next_state['nrecproms'],
                                                   next_state['nrecgifts'],
                                                   next_state['totrecamt'],
                                                   next_state['recamtpergift'],
                                                   next_state['recamtperprom'],
                                                   next_state['promrecency'],
                                                   next_state['timelag'],
                                                   next_state['recencyratio'],
                                                   next_state['promrecratio'],
                                                   next_state['respondedbit1'],
                                                   next_state['respondedbit2'],
                                                   next_state['respondedbit3'],
                                                   next_state['mailedbit1'],
                                                   next_state['mailedbit2'],
                                                   next_state['mailedbit3']
                                                   ])
                training_data['done'] = str(i == 15)
                
                csv_writer.writerow(training_data)

            row_num += 1


def main(args):
    input_data_file = "data/input_data.csv"
    output_data_file = "data/training_data.csv"

    process_input_data(input_data_file, output_data_file)


if __name__ == '__main__':
    main(sys.argv)

    sys.exit(0)
