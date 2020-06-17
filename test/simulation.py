import sys
import numpy as np
import time
import os
import filecmp
import datetime
import random
import argparse

sys.path.append("../dnam/")
from processfile import ProcessFile
from log import get_logger

parser = argparse.ArgumentParser(description="Simulation file for dNAM")
parser.add_argument("-s", "--start", help="Starting file size in bytes", type=int, default=20)
parser.add_argument("-e", "--end", help="Ending file size in bytes", type=int, default=2000)
parser.add_argument("-g", "--gap", help="Gap between file size", type=int, default=40)
parser.add_argument("-ec", "--error_check", help="Maxiumum number of error that will be checked for each origami",
                    type=int, default=8)
parser.add_argument("-oc", "--copies_of_each_origami", help="A single origami will have multiple copy with different"
                                                            " level of error", type=int, default=10)
parser.add_argument("-fp", "--false_positive_per_origami", help="Number of false positive error that will be "
                                                                "added in each origami", type=int, default=0)
parser.add_argument("-a", "--average", help="The error will be distributed on average per origami", action="store_true")
parser.add_argument("-v", "--verbose", help="Print all the details", type=int, default=0)
args = parser.parse_args()

STARTING_FILE_SIZE = args.start  # Starting file size in bytes for simulation
ENDING_FILE_SIZE = args.end  # Ending file size in bytes for simulation
FILE_SIZE_GAP = args.gap  # Interval size of the file for simulation
MAXIMUM_NUMBER_OF_ERROR_CHECKED_PER_ORIGAMI = args.error_check
MAXIMUM_NUMBER_OF_ERROR_INSERTED_PER_ORIGAMI = args.error_check
COPIES_OF_EACH_ORIGAMIES = args.copies_of_each_origami
FALSE_POSITIVE_PER_ORIGAMI = args.false_positive_per_origami
AVERAGE = args.average
VERBOSE = args.verbose

RESULT_DIRECTORY = "test_result"
DATA_DIRECTORY = "test_data"
CURRENT_DATE = str(datetime.datetime.now()).replace("-", "")[:8]
SIMULATION_DIRECTORY = RESULT_DIRECTORY + "/" + CURRENT_DATE + "_simulation_" + str(STARTING_FILE_SIZE) + "_" + str(
    ENDING_FILE_SIZE)

logger = get_logger(VERBOSE, __name__)


def change_orientation(origamies):
    """
    This will randomly alter the origami orientation
    :param origamies:
    :return:
    """
    for i, single_origami in enumerate(origamies):
        orientation_option = random.choice(range(4))
        # orientation_option = 0
        single_matrix = dnam_object.data_stream_to_matrix(single_origami.rstrip('\n'))
        if orientation_option == 0:
            single_matrix = single_matrix
        elif orientation_option == 1:
            single_matrix = np.flipud(single_matrix)
        elif orientation_option == 2:
            single_matrix = np.fliplr(single_matrix)
        else:
            single_matrix = np.fliplr(np.flipud(single_matrix))
        origamies[i] = single_matrix
    logger.info("Origami has been randomly oriented")
    return origamies


def degrade(dnam_object, file_in, file_out, number_of_error, average=False, false_positive_per_origami=2, copies=10):
    try:
        degrade_file = open(file_out, "w")
        encoded_file = open(file_in, "r")
    except Exception as e:
        logger.error(e)
    total_error_inserted = 0
    origami_number = 0
    encode_file_list = encoded_file.readlines() * copies
    random.shuffle(encode_file_list)
    encode_file_list = change_orientation(encode_file_list)
    false_positive_details = {}
    total_origami = len(encode_file_list)
    total_error = int(number_of_error * total_origami)
    total_maximum_false_positive_insert = int(total_error * .3)
    false_positive_inserted = 0
    # Error will be introduced on average per origami
    if average:
        # Each origami will have different number of error
        # Total number of error will be same
        while total_error_inserted < total_error:
            random_index = random.choice(range(total_origami))
            if random_index not in false_positive_details:
                false_positive_details[random_index] = 0
            error_row = random.choice(range(6))
            error_column = random.choice(range(8))
            if encode_file_list[random_index][error_row][error_column] == 1:
                encode_file_list[random_index][error_row][error_column] = 0
                total_error_inserted += 1
            elif false_positive_details[random_index] < false_positive_per_origami \
                    and false_positive_inserted < total_maximum_false_positive_insert:
                encode_file_list[random_index][error_row][error_column] = 1
                false_positive_details[random_index] += 1
                total_error_inserted += 1
                false_positive_inserted += 1
    # Each origami will be fixed amount of error
    else:
        # each origami will have same number of error
        for index in range(len(encode_file_list)):
            error_inserted = 0
            if index not in false_positive_details:
                false_positive_details[index] = 0
            while error_inserted < number_of_error:  # In error only 1 will be 0. 0 won't be 1
                error_row = random.choice(range(6))
                error_column = random.choice(range(8))
                if encode_file_list[index][error_row][error_column] == 1:
                    encode_file_list[index][error_row][error_column] = 0
                    total_error_inserted += 1
                    error_inserted += 1
                elif false_positive_details[index] < false_positive_per_origami and \
                        false_positive_inserted < total_maximum_false_positive_insert:
                    encode_file_list[index][error_row][error_column] = 1
                    false_positive_details[index] += 1
                    total_error_inserted += 1
                    false_positive_inserted += 1
                    error_inserted += 1

    for single_origami in encode_file_list:
        degraded_origami = dnam_object.matrix_to_data_stream(single_origami)
        degrade_file.write(degraded_origami + "\n")
        origami_number += 1
    degrade_file.close()
    encoded_file.close()
    logger.info("Error insertion done")
    return total_error_inserted


# Create the simulation directory if it's not there already
if not os.path.isdir(SIMULATION_DIRECTORY):
    os.makedirs(SIMULATION_DIRECTORY)
# Result file name
result_file_name = SIMULATION_DIRECTORY + "/overall_result_simulation.csv"
# create result file
try:
    # If file exists we will just append the result otherwise create new file
    if os.path.isfile(result_file_name):
        result_file = open(result_file_name, "a")
    else:
        result_file = open(result_file_name, "w")
        result_file.write("File size,Bits per origami,Redundancy,Origami without redundancy,Origami with redundancy,"
                          "Number of copies of each origami,Encoding time,"
                          "Number of error per origami,Total number of error,Total number of error detected,"
                          "Incorrect origami,Correct Origami,Missing origamies,Decoding time,status,threshold data,"
                          "threshold parity,false positive\n")
    # Closing the file otherwise it will be copied on the multi processing
    # Each process copied all the open file descriptor. If this we don't close this file here.
    # Then this file descriptor  will be copied by each process hence memory consumption will
    # be more / we might get the error of too many file open
    # So we will close this here before we call the multiprocessing
    result_file.close()
except Exception as e:
    logger.error("Couldn't open the result file")
    logger.exception(e)
    exit()

for file_size in list(range(STARTING_FILE_SIZE, ENDING_FILE_SIZE, FILE_SIZE_GAP)):
    test_file_name = SIMULATION_DIRECTORY + "/test_" + str(file_size)
    logger.info("working with file size: {file_size}".format(file_size=file_size))
    # Generate random binary file for encoding
    with open(test_file_name, "wb", 0) as random_file:
        random_file.write(os.urandom(file_size))
    # encode the randomly generated file
    dnam_object = ProcessFile(redundancy=50, verbose=False)
    encoded_file_name = test_file_name + "_encode"
    decoded_file_name = test_file_name + "_decode"
    start_time = time.time()
    # Encode the file
    segments, droplet, data_bit_per_origami, required_red = dnam_object.encode(test_file_name, encoded_file_name)
    encoding_time = round((time.time() - start_time), 2)
    for error_in_each_origami in range(MAXIMUM_NUMBER_OF_ERROR_CHECKED_PER_ORIGAMI + 1):
        error_in_each_origami = round(error_in_each_origami, 2)
        logger.info("Checking error: {error_in_each_origami}".format(error_in_each_origami=error_in_each_origami))
        degraded_file_name = encoded_file_name + "_degraded_copy_" + str(
            COPIES_OF_EACH_ORIGAMIES) + "_error_" + str(error_in_each_origami)
        decoded_file_name = test_file_name + "_decoded_copy_" + str(COPIES_OF_EACH_ORIGAMIES) + "_error_" + str(
            error_in_each_origami)
        # if error_in_each_origami == 0:
        total_error_insertion = degrade(dnam_object=dnam_object, file_in=encoded_file_name,
                                        file_out=degraded_file_name,
                                        number_of_error=error_in_each_origami, average=AVERAGE,
                                        false_positive_per_origami=FALSE_POSITIVE_PER_ORIGAMI,
                                        copies=COPIES_OF_EACH_ORIGAMIES)
        logger.info("Degradation done")
        dnam_decode = ProcessFile(redundancy=50, verbose=VERBOSE)
        # try to decode with different decoding parameter
        for threshold_data in range(2, 3):  # This two loops are for the parameter.
            for threshold_parity in range(2, 3):  # Now we are choosing only one parameter.
                decoded_file_name = test_file_name + "_decoded_copy_" + str(COPIES_OF_EACH_ORIGAMIES) + "_error_" + \
                                    str(error_in_each_origami) + "_scp_" + str(threshold_data) + \
                                    "_tempweight_" + str(threshold_parity)
                start_time = time.time()
                try:
                    decoding_status, incorrect_origami, correct_origami, total_error_fixed, missing_origamies \
                        = dnam_decode.decode(degraded_file_name, decoded_file_name, file_size,
                                             threshold_data=threshold_data,
                                             threshold_parity=threshold_parity,
                                             maximum_number_of_error=MAXIMUM_NUMBER_OF_ERROR_CHECKED_PER_ORIGAMI,
                                             false_positive=FALSE_POSITIVE_PER_ORIGAMI,
                                             individual_origami_info=True,
                                             correct_file=encoded_file_name)
                    if os.path.exists(encoded_file_name) and os.path.exists(decoded_file_name) and filecmp.cmp(
                            test_file_name, decoded_file_name):
                        status = 1
                    else:
                        if decoding_status == -1:
                            status = -1  # We could detect
                        else:
                            status = 0  # We couldn't detect
                            print("Couldn't detect")
                except Exception as e:
                    print(e)  # Something went wrong on the decoding
                    status = -2
                    incorrect_origami = -1
                    correct_origami = -1
                    total_error_fixed = -1
                    missing_origami = []
                decoding_time = round((time.time() - start_time), 2)
                with open(result_file_name, "a") as result_file:
                    result_file.write("{FILE_SIZE},{bits_per_origami},{redundancy},{total_origami_without_red},{total_origami_with_red},{copy},\
                        {encoding_time},{error_in_each_origami},{total_error_insertion},{total_error_fixed},\
                        {incorrect_origami},{correct_origami},{missing_oirgami},{decoding_time},{status},{single_threshold_data},\
                        {single_threshold_parity},{false_positive}\n".format(
                        FILE_SIZE=file_size,
                        bits_per_origami=data_bit_per_origami,
                        redundancy=required_red,
                        total_origami_without_red=segments,
                        total_origami_with_red=droplet,
                        copy=COPIES_OF_EACH_ORIGAMIES,
                        encoding_time=encoding_time,
                        error_in_each_origami=error_in_each_origami,
                        total_error_insertion=total_error_insertion,
                        total_error_fixed=total_error_fixed,
                        incorrect_origami=incorrect_origami,
                        correct_origami=correct_origami,
                        missing_oirgami=str(missing_origamies).replace(",", " "),
                        decoding_time=decoding_time,
                        status=status,
                        single_threshold_data=threshold_data,
                        single_threshold_parity=threshold_parity,
                        false_positive=FALSE_POSITIVE_PER_ORIGAMI
                    ))
                if os.path.exists(decoded_file_name) and status == 1:
                    os.remove(decoded_file_name)
                    pass
        del dnam_decode
        if status == 1:  # if we can decode the file we will remove that. Otherwise keep it for future debug reference.
            os.remove(degraded_file_name)
    del dnam_object  # clearing up the memory
