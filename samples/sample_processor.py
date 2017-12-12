import argparse
import csv
import os
import sys

from datetime import datetime

DEFAULT_MAILING_COST = 0.68


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

        # write out the training data file




def main(args):
    pass

if __name__ == '__main__':
    main(sys.argv)

    sys.exit(0)
