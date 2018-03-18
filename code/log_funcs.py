"""
MNIST Numpy Assignments
Advanced Topics in Machine Learning, UCL (COMPGI13)
Neural Network Classes
Author: Adam Hornsby
"""
import sys
import os
import csv

def _check_log_directory(directory):
    """
    Check that the log directory exists and create it if it doesn't
    """
    try:
        if not os.path.exists(directory):
            print "Attempting to make log directory at " + directory
            os.makedirs(directory)
    except IOError as e:
        sys.exit("Error attempting to create log directory: {0}".format(e.strerror).strip())


def _initialise_model_log(log_filepath):
    """Create a model logging file if it doesn't already exist"""
    if not os.path.exists(log_filepath):
        with open(log_filepath, 'a') as fp:
            a = csv.writer(fp, delimiter=',')
            data = [['DATETIME', 'Model ID', 'Model', 'Parameters', 'Performance', 'Files']]
            a.writerows(data)


def log_model_results(log_name, log_dir, datetime, model_id, model, parameters, performance, files):
    """Log model performance to a file"""

    filepath = str(log_dir) + '/' + str(log_name) + '.csv'

    # check logging directory
    _check_log_directory(log_dir)

    # initialise file if it doesn't already exist
    _initialise_model_log(filepath)

    # add model results to file
    with open(filepath, 'a') as fp:
        a = csv.writer(fp, delimiter=',')
        data = [[datetime, model_id, model, parameters, performance, files]]
        a.writerows(data)