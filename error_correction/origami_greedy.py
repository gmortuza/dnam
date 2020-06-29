from functools import reduce
from collections import Counter
import copy
import numpy as np
import logging
from log import get_logger


class Origami:
    """
    This class handle individual origami. Both the encoding and decoding is handled by this class. This class is
    inherited by ProcessFile class. The ProcessFile class calls/handel all the method of this class. Each origami is
    represented by a matrix. So the term matrix and origami is used interchangeably.
    """

    def __init__(self, verbose=0):
        """
        :param verbose: is it running on debug mode or info mode
        """
        self.row = 6
        self.column = 8
        self.checksum_bit_per_origami = 4
        self.encoded_matrix = None
        self.recovered_matrix_info = []
        self.list_of_error_combination = []
        self.orientation_details = {
            '0': 'Orientation of the origami is correct',
            '1': 'Origami was flipped in horizontal direction',
            '2': 'Origami was flipped in vertical direction.',
            '3': 'Origami was flipped in both direction. '
        }
        self.logger = get_logger(verbose, __name__)

    @staticmethod
    def _matrix_details(data_bit_per_origami: int) -> object:
        """
        Returns the relationship of the matrix. Currently all the the relationship is hardcoded.
        This method returns the following details:
            parity bits: 20 bits
            indexing bits: depends on the file size
            orientation bits: 4 bits
            checksum bits: 4 bits
            data bits: 48 - indexing bits
        :rtype: object
        :param data_bit_per_origami: Number of bits that will be encoded in each origami
        :returns matrix_details -> Label for each cell
                 parity_bit_relation: Parity bit mapping
                 checksum_bit_relation: Checksum bit mapping

        """

        # Data following indices of the matrix contains the data and indexes of the origami
        data_index = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (2, 7), (3, 7),
                      (5, 7), (5, 6), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0), (3, 0), (2, 0)]
        parity_bit_relation = {
            (1, 1): [(0, 0), (0, 1), (0, 5), (2, 3), (5, 4)],
            (1, 2): [(0, 1), (0, 3), (1, 0), (2, 4), (2, 7)],
            (1, 3): [(0, 0), (0, 2), (0, 4), (3, 7), (5, 2)],
            (1, 4): [(0, 3), (0, 5), (0, 7), (3, 0), (5, 5)],
            (1, 5): [(0, 4), (0, 6), (1, 7), (2, 0), (2, 3)],
            (1, 6): [(0, 2), (0, 6), (0, 7), (2, 4), (5, 3)],
            (2, 1): [(1, 0), (2, 3), (3, 0), (4, 7), (5, 0), (5, 3)],
            (2, 6): [(1, 7), (2, 4), (3, 7), (4, 0), (5, 4), (5, 7)],
            (3, 1): [(0, 0), (0, 3), (1, 7), (2, 0), (3, 3), (4, 0)],
            (3, 6): [(0, 4), (0, 7), (1, 0), (2, 7), (3, 4), (4, 7)],
            (4, 1): [(0, 4), (3, 3), (5, 0), (5, 1), (5, 5)],
            (4, 2): [(3, 4), (3, 7), (4, 0), (5, 1), (5, 3)],
            (4, 3): [(0, 2), (2, 7), (5, 0), (5, 2), (5, 4)],
            (4, 4): [(0, 5), (2, 0), (5, 3), (5, 5), (5, 7)],
            (4, 5): [(3, 0), (3, 3), (4, 7), (5, 4), (5, 6)],
            (4, 6): [(0, 3), (3, 4), (5, 2), (5, 6), (5, 7)],
            (2, 2): [(0, 2), (1, 0), (2, 0), (5, 6), (5, 0)],
            (2, 5): [(0, 5), (1, 7), (2, 7), (5, 1), (5, 7)],
            (3, 2): [(0, 6), (3, 0), (4, 0), (5, 2), (0, 0)],
            (3, 5): [(0, 1), (3, 7), (4, 7), (5, 5), (0, 7)]
        }

        matrix_details = dict(
            data_bits=data_index[:data_bit_per_origami],
            orientation_bits=[(1, 0), (1, 7), (4, 0), (4, 7)],
            indexing_bits=data_index[data_bit_per_origami:],
            checksum_bits=[(2, 3), (2, 4), (3, 3), (3, 4)],
            orientation_data=[1, 1, 1, 0]
        )
        checksum_bit_relation = {
            (2, 3): [(0, 0), (0, 2), (1, 7), (2, 0), (5, 3), (5, 6)],
            (2, 4): [(0, 5), (0, 7), (1, 0), (2, 7), (5, 1), (5, 4)],
            (3, 3): [(0, 3), (0, 6), (3, 0), (4, 7), (5, 0), (5, 2)],
            (3, 4): [(0, 1), (0, 4), (3, 7), (4, 0), (5, 5), (5, 7)]
        }

        return matrix_details, parity_bit_relation, checksum_bit_relation

    def create_initial_matrix_from_binary_stream(self, binary_stream: str, index: int) -> object:
        """
        Insert droplet data, orientation and index bit in the matrix
        :param binary_stream: Binary data that need to be encoded in the matrix
        :param index: Indexing of the matrix
        :return: data_matrix: Matrix with droplet data, index and orientation bits
        """
        binary_list = list(binary_stream)
        data_matrix = np.full((self.row, self.column), -1)  # All the cell of the matrix will have initial value -1.

        # Putting the data into matrix
        for i, bit_index in enumerate(self.matrix_details["data_bits"]):
            data_matrix[bit_index[0]][bit_index[1]] = binary_list[i]

        # Putting orientation data
        for i, bit_index in enumerate(self.matrix_details["orientation_bits"]):
            data_matrix[bit_index[0]][bit_index[1]] = self.matrix_details['orientation_data'][i]

        # Putting indexing bits.
        # Checking if current index is more than supported.
        if index >= 2 ** len(self.matrix_details["indexing_bits"]):
            self.logger.error(
                'Maximum support index is {maximum_input}. But given index is {index}'.format(
                    maximum_input=2 ** len(self.matrix_details["indexing_bits"]),
                    index=index
                ))
            raise ValueError("Maximum support of index exceed")
        index_len = '0' + str(len(self.matrix_details["indexing_bits"])) + 'b'
        index_bin = list(format(index, index_len))
        # Set the indexing
        for i, bit_index in enumerate(self.matrix_details["indexing_bits"]):
            data_matrix[bit_index[0]][bit_index[1]] = index_bin[i]

        self.logger.info("Droplet data and index has been inserted")
        return data_matrix

    @staticmethod
    def _xor_matrix(matrix, relation):
        """
        XOR the data using the relation
        :param matrix: Matrix upon which XOR will be implemented
        :param relation: mapping of the XOR data. May contain checksum/parity mapping.
        :return: matrix: Matrix with the XOR value
        """
        for single_xor_relation in relation:
            # Getting the all the data bits related to a specific parity bit
            data_bits_value = [int(matrix[a[0]][a[1]]) for a in
                               relation[single_xor_relation]]
            # XORing all the data bits
            xored_value = reduce(lambda i, j: int(i) ^ int(j), data_bits_value)
            # Update the parity bit with the XORed value
            matrix[single_xor_relation[0]][single_xor_relation[1]] = int(xored_value)

        return matrix

    def _encode(self, binary_stream, index, data_bit_per_origami):
        """
        Handle the encoding. Most of the time handle xoring.
        :param binary_stream: Binary value of the data
        :param index: Index of the current matrix
        :param data_bit_per_origami: Number of bits that will be encoded in each origami
        :return: Encoded matrix
        """
        # Create the initial matrix which will contain the word,index and binary bits for fixing orientation but no
        # error encoding. So the parity bits will have the initial value of -1
        self.number_of_bit_per_origami = data_bit_per_origami
        self.matrix_details, self.parity_bit_relation, self.checksum_bit_relation = \
            self._matrix_details(data_bit_per_origami)
        self.data_bit_to_parity_bit = Origami.data_bit_to_parity_bit(self.parity_bit_relation)
        encoded_matrix = self.create_initial_matrix_from_binary_stream(binary_stream, index)

        # Set the cell value in checksum bits. This has to be before the parity bit xoring. Cause the parity bit
        # contains the checksum bits. And the default value of the checksum bit is -1. So if the parity xor happens
        # before checksum xor then some of the parity bit will have value negative. as that would be xor with -1
        encoded_matrix = Origami._xor_matrix(encoded_matrix, self.checksum_bit_relation)
        self.logger.info("Finish calculating the checksum")
        # XOR for the parity code
        encoded_matrix = Origami._xor_matrix(encoded_matrix, self.parity_bit_relation)
        self.logger.info("Finish calculating the parity bits")
        return Origami.matrix_to_data_stream(encoded_matrix)

    def encode(self, binary_stream, index, data_bit_per_origami):
        """
        This method will not be called internally. This is added just for testing purpose

        :param binary_stream: Binary value of the data
        :param index: Index of the current matrix
        :param data_bit_per_origami: Number of bits that will be encoded in each origami
        :return: Encoded matrix
        :return:
        """
        return self._encode(binary_stream, index, data_bit_per_origami)

    @staticmethod
    def data_bit_to_parity_bit(parity_bit_relation):
        """
        Reverse the parity bit to data bit.
        :param parity_bit_relation: A dictionary that contains parity bit as key and
         the respective indices that will be XORed in the parity bit as value.
        :return: data_bit_to_parity_bit: A dictionary that contains indices as key and and respective parity
         indices that used that indices for XORing.
        """
        data_bit_to_parity_bit = {}
        for single_parity_bit in parity_bit_relation:
            # Loop through each parity bit relation and add those
            for single_data_bit in parity_bit_relation[single_parity_bit]:
                data_bit_to_parity_bit.setdefault(single_data_bit, []).append(single_parity_bit)
        return data_bit_to_parity_bit

    def show_encoded_matrix(self):
        """
        Display encoded matrix

        :returns: None
        """
        self.print_matrix(self.encoded_matrix)

    @staticmethod
    def print_matrix(matrix, in_file=False):
        """
        Display a given matrix

        :param: matrix: A 2-D matrix
        :param: in_file: if we want to save the encoding information in a file.

        :returns: None
        """
        for row in range(len(matrix)):
            for column in range(len(matrix.T)):
                if not in_file:
                    print(matrix[row][column], end="\t")
                else:
                    print(matrix[row][column], end="\t", file=in_file)
            if not in_file:
                print("")
            else:
                print("", file=in_file)

    @staticmethod
    def matrix_to_data_stream(matrix):
        """
        Convert 2-D matrix to string

        :param: matrix: A 2-D matrix
        :returns: data_stream: string of 2-D matrix
        """
        data_stream = []
        for row in range(len(matrix)):
            for column in range(len(matrix.T)):
                data_stream.append(matrix[row][column])
        return ''.join(str(i) for i in data_stream)

    def data_stream_to_matrix(self, data_stream):
        """
        Convert a sting to 2-D matrix

        The length of data stream should be 48 bit currently this algorithm is only working with 6x8 matrix

        :param: data_stream: 48 bit of string
        :returns: matrix: return 2-D matrix
        """
        matrix = np.full((self.row, self.column), -1)
        data_stream_index = 0
        for row in range(len(matrix)):
            for column in range(len(matrix.T)):
                matrix[row][column] = data_stream[data_stream_index]
                data_stream_index += 1
        return matrix

    def _fix_orientation(self, matrix, option=0):
        """
        Fix the orientation of the decoded matrix. Option parameter will decide which way matrix will be tested now.
        Initially we will check the default matrix(as it was passed). Later we will called this method recursively
        and increase the option value. If option value is 3 and the orientation doesn't match then we will mark this
        origami as not fixed.

        First option is using current matrix
        Second option is reversing the current matrix that will fix the vertically flipped issue
        Third option is mirroring the current matrix that will fix the horizontally flipped issue
        Fourth option is both reverse then mirror the current matrix that will fix
        both vertically flipped and horizontally flipped issue

        :param: matrix: Decoded matrix
                option: On which direction the matrix will be flipped now

        Returns:
            matrix: Orientation fixed matrix.
        """

        if option == 0:
            corrected_matrix = matrix
        elif option == 1:
            # We will just take the reverse/Flip in horizontal direction
            corrected_matrix = np.flipud(matrix)
        elif option == 2:
            # We will take the mirror/flip in vertical direction
            corrected_matrix = np.fliplr(matrix)
        elif option == 3:
            # Flip in both horizontal and vertical direction
            corrected_matrix = np.flipud(np.fliplr(matrix))
        else:
            # The orientation couldn't be determined
            # This is not correctly oriented. Will remove that after testing
            self.logger.info("Couldn't orient the origami")
            return -1, matrix
        orientation_check = True
        for i, bit_index in enumerate(self.matrix_details["orientation_bits"]):
            if corrected_matrix[bit_index[0]][bit_index[1]] != self.matrix_details["orientation_data"][i]:
                orientation_check = False
        if orientation_check:
            # returning option will tell us which way the origami was oriented.
            self.logger.info("Origami was oriented successfully")
            return option, corrected_matrix
        else:
            # Matrix isn't correctly oriented so we will try with other orientation
            return self._fix_orientation(matrix, option + 1)

    def _find_possible_error_location(self, matrix):
        """
        Return all the correct and incorrect parity bits.

        :param: matrix: 2-D matrix
        :returns: correct_indexes: Indices of all correct parity bits
                  incorrect_indexes: Indices of all incorrect parity bit
        """
        correct_indexes = []
        incorrect_indexes = []
        for parity_bit_index in self.parity_bit_relation:
            # Now xoring every element again and checking it's correct or not
            nearby_values = [int(matrix[a[0]][a[1]]) for a in self.parity_bit_relation[parity_bit_index]]
            xored_value = reduce(lambda i, j: int(i) ^ int(j), nearby_values)
            if matrix[parity_bit_index[0]][parity_bit_index[1]] == int(xored_value):
                correct_indexes.append(parity_bit_index)
            else:
                incorrect_indexes.append(parity_bit_index)
        return correct_indexes, incorrect_indexes

    def _is_matrix_correct(self, matrix):
        """
        Check if all the bits of the matrix are correct or not

        Parameter:
            matrix: A 2-D matrix
        Returns:
            Boolean: True if matrix is correct false otherwise
        """
        correct_indexes, incorrect_indexes = self._find_possible_error_location(matrix)
        return len(incorrect_indexes) == 0

    def _decode(self, matrix, threshold_parity, threshold_data,
                maximum_number_of_error, false_positive):
        """
        :param matrix: Matrix that will be decoded
        :param threshold_parity: Threshold value of parity cell to be considered as error
        :param threshold_data: Threshold value of data cell to be considered as error
        :param maximum_number_of_error: Maximum number of error that will be checked
        :param false_positive: Number of false positive that will be checked
        :return:
        """
        # We will try to decode multiple origami so we are making the variable empty at the first
        matrix_details = {}
        # Will check the matrix weight first.
        # If matrix weight is zero that means all the parity matched
        # We will reduce this matrix weight by a greedy approach
        _, matrix_weight, probable_error = self._get_matrix_weight(matrix, [], threshold_parity,
                                                                   threshold_data, false_positive)
        if matrix_weight == 0:
            # All parity matched now we will check orientation and checksum
            self.logger.info("No parity mismatch found at the first step")
            single_recovered_matrix = self.return_matrix(matrix, [])
            if not single_recovered_matrix == -1:
                # If we don't get -1 then it means orientation and checksum also matched
                # So we will return the recovered matrix
                return single_recovered_matrix
        # We will alter each of the probable error one at a time and recalculate the matrix weight
        for single_error in probable_error:
            matrix_details[tuple(single_error)] = {}
            changed_matrix, matrix_details[tuple(single_error)]["error_value"], \
                matrix_details[tuple(single_error)]["probable_error"] = \
                self._get_matrix_weight(matrix, [single_error], threshold_parity, threshold_data,
                                        false_positive)
            # If after altering one bit only matrix_weight becomes zero then we will check checksum and parity
            if matrix_details[tuple(single_error)]["error_value"] == 0:
                self.logger.info("After altering one bit, all the parity matched")
                single_recovered_matrix = self.return_matrix(changed_matrix, [single_error])
                if not single_recovered_matrix == -1:
                    return single_recovered_matrix
        # We will sort the matrix based on the matrix weight
        matrix_details = {k: v for k, v in sorted(matrix_details.items(), key=lambda item: item[1]["error_value"])}

        for single_error in matrix_details:
            error_combination_checked_so_far = [single_error]
            # We will alter all the probable error list which we got during calculating the matrix weight
            errors_that_will_be_checked = matrix_details[single_error]["probable_error"]
            # This queue will contains which cell to alter next
            queue_for_single_error = {}
            # we will not check more than the maximum number of error
            while len(error_combination_checked_so_far) < maximum_number_of_error and len(
                    errors_that_will_be_checked) >= 1:
                # Contains all the matrix weights. Which will be sorted to choose next bit flip
                matrix_weights = {}
                for single_error_in_probable_error in errors_that_will_be_checked:
                    will_check_now = error_combination_checked_so_far + [single_error_in_probable_error]
                    # Find matrix error after altering will_check_now
                    changed_matrix, single_probable_matrix_weight, single_probable_error = self._get_matrix_weight(
                        matrix,
                        will_check_now,  # Alter this one
                        threshold_parity,
                        threshold_data,
                        false_positive)

                    if single_probable_matrix_weight == 0:
                        single_recovered_matrix = self.return_matrix(changed_matrix, will_check_now)
                        if not single_recovered_matrix == -1:
                            return single_recovered_matrix

                    # Add this newly gotten matrix weight to the variable which contains all the matrix weights
                    if single_probable_matrix_weight in matrix_weights:
                        # This error value is already there so we will append the new value in the list
                        matrix_weights[single_probable_matrix_weight]["probable_error"].append(single_probable_error)
                        matrix_weights[single_probable_matrix_weight]["cell_checked_so_far"].append(
                            tuple(will_check_now))
                    else:
                        matrix_weights[single_probable_matrix_weight] = {}
                        matrix_weights[single_probable_matrix_weight]["probable_error"] = [single_probable_error]
                        matrix_weights[single_probable_matrix_weight]["cell_checked_so_far"] = [tuple(will_check_now)]
                try:
                    # We will keep just lowest two matrix weight in the queue
                    minimum_error_values = sorted(matrix_weights.keys())[0:2]
                except:  # matrix_weights doesn't have two entry. so will get a key exception
                    minimum_error_values = [min(sorted(matrix_weights.keys()))]
                self.logger.info("current matrix weight: " + str(minimum_error_values[0]) + " after altering the cell: "
                                 + str(matrix_weights[minimum_error_values[0]]["cell_checked_so_far"]))
                # Iter over the number of items that are in this minimum error value and make a queue from that
                # making this queue cause some of the minimum error value will have more than one indexes
                for minimum_error_value in minimum_error_values:
                    for i in range(len(matrix_weights[minimum_error_value]["probable_error"])):
                        queue_for_single_error[matrix_weights[minimum_error_value]["cell_checked_so_far"][i]] = \
                            matrix_weights[minimum_error_value]["probable_error"][i]
                for i in sorted(queue_for_single_error, key=len, reverse=True):
                    error_combination_checked_so_far = list(i)
                    errors_that_will_be_checked = set(queue_for_single_error[i]).difference(
                        set(error_combination_checked_so_far))
                    del queue_for_single_error[i]
                    if len(error_combination_checked_so_far) < maximum_number_of_error:
                        break
                    else:
                        continue
        # If it comes to this point that means we were not able return a correct matrix so far.
        # And we don't have any options
        return -1

    def return_matrix(self, correct_matrix, error_locations):
        """
        This will check the orientation and checksum of the matrix.
        If the orientation is correct then droplet data and index will be extracted.
        If orientation cannot be correct then -1 will be return

        :param correct_matrix: Matrix with no error
        :param error_locations: Error location that has been fixed
        :return: single_recovered_matrix: Dictionary which contains details of an individual origami
        """
        # will return this dictionary which will have all the origami details
        single_recovered_matrix = {}
        orientation_info, correct_matrix = self._fix_orientation(correct_matrix)

        if not orientation_info == -1 and self.check_checksum(correct_matrix):
            single_recovered_matrix['orientation_details'] = self.orientation_details[
                str(orientation_info)]
            single_recovered_matrix['orientation'] = orientation_info
            single_recovered_matrix['matrix'] = correct_matrix
            single_recovered_matrix['orientation_fixed'] = True
            single_recovered_matrix['total_probable_error'] = len(error_locations)
            single_recovered_matrix['probable_error_locations'] = error_locations
            single_recovered_matrix['is_recovered'] = True
            single_recovered_matrix['checksum_checked'] = True
            single_recovered_matrix['index'], single_recovered_matrix[
                'binary_data'] = \
                self._extract_text_and_index(correct_matrix)
            self.logger.info("Origami error fixed. Error corrected: " + str(error_locations))
            return single_recovered_matrix
        else:  # If orientation or checksum doesn't match we will return -1
            self.logger.info("Orientation/checksum didn't match")
            return -1

    def _extract_text_and_index(self, matrix):
        """
        Get droplet data and index of the droplet from the origami
        :param matrix: Matrix from where information will be extracted.
        :return: (index, droplet_data)
        """
        if matrix is None:
            return
        # Extracting index first
        index_bin = []
        for bit_index in self.matrix_details['indexing_bits']:
            index_bin.append(matrix[bit_index[0]][bit_index[1]])
        index_decimal = int(''.join(str(i) for i in index_bin), 2)
        # Extracting the text now
        # Extracting text index
        text_bin_data = ""
        for bit_index in self.matrix_details['data_bits']:
            text_bin_data += str(matrix[bit_index[0]][bit_index[1]])

        return index_decimal, text_bin_data

    def _get_matrix_weight(self, matrix, changing_location, threshold_parity, threshold_data,
                           false_positive):
        """
        Matrix weight indicates how much error does this individual matrix contains.
        More the matrix error is more error this matrix contains.
        We will use a greedy approach to reduce this matrix error.

        :param matrix: Whose weight will be calculated
        :param changing_location: Locations of the matrix that will be changed before calculating the weight
        :param threshold_parity: Threshold value for consider a parity bit to have an error
        :param threshold_data: Threshold value for consider a data bit to have an error
        :param false_positive: Will we check false positive or not
        :return:
        """
        # We will change few bits (based on changing_locaiton parameter).
        # If we don't make a deep copy it will modify the original matrix that was passed.
        matrix_copy = copy.deepcopy(matrix)
        total_false_positive_added = 0  # False positive added from data only
        false_positive_added_in_parity = 0  # False positive added from parity only
        # Will filp the bit
        for single_changing_location in changing_location:
            if matrix_copy[single_changing_location[0]][single_changing_location[1]] == 0:
                matrix_copy[single_changing_location[0]][single_changing_location[1]] = 1
            else:
                if single_changing_location in self.parity_bit_relation:
                    false_positive_added_in_parity += 1
                else:
                    total_false_positive_added += 1
                matrix_copy[single_changing_location[0]][single_changing_location[1]] = 0
        # Check which parity bit matched and which didn't
        parity_bit_indexes_correct, parity_bit_indexes_incorrect = self._find_possible_error_location(
            matrix_copy)
        # Marking all the indices that is related to the incorrect parity bit
        probable_error_indexes = [j for i in parity_bit_indexes_incorrect for j in self.parity_bit_relation[i]]
        # Marking all the indexes that is related to the unmatched checksum
        probable_error_from_checksum = []
        # Checksum indexes that didn't match after xoring it's related indices.
        unmatched_checksum = []
        for single_checksum_index, single_checksum_relation in self.checksum_bit_relation.items():
            nearby_values = [int(matrix_copy[a[0]][a[1]]) for a in single_checksum_relation]
            xored_value = reduce(lambda i, j: int(i) ^ int(j), nearby_values)
            # As it didn't match it might have some error
            if matrix_copy[single_checksum_index[0]][single_checksum_index[1]] != xored_value:
                unmatched_checksum.append(single_checksum_index)
                probable_error_indexes.append(single_checksum_index)
                probable_error_from_checksum.extend(
                    single_checksum_relation)  # This will be the second parameter to order the error list
        # probable_error_index = [item for item in Counter(probable_error_indexes).most_common() if item[1] >= 3]
        # If the error also present in the probable checksum error then
        # we will increase the number of occurrence of these index
        # All Probable error indexes expect the checksum after checking the temporary weight.
        # Will contain { temporary_weight: [all indexes of same temporary weight] }
        probable_data_error = {}
        # Final parity error. After checking the temporary weight
        probable_parity_error = []
        # Parity bit error that is taken from error data bit
        probable_parity_error_all = []
        # Putting the temporary weight on each data and checksum indexes.
        # The default temporary weight will be it's number of occurrence from unmatched parity error.
        for item in Counter(probable_error_indexes).most_common():
            if item[0] in probable_error_from_checksum and item[0] in unmatched_checksum:
                temp_weight = item[1] + 2
            elif item[0] in probable_error_from_checksum:
                temp_weight = item[1] + 1
            elif item[0] in unmatched_checksum:
                temp_weight = item[1] + 1
            else:
                temp_weight = item[1]
            # threshold_parity = item[1] + 1 if item[0] in probable_error_from_checksum else item[1]
            # if threshold_parity < threshold_data:
            #     break
            probable_data_error.setdefault(temp_weight, []).append(item[0])
            probable_parity_error_all.extend(self.data_bit_to_parity_bit[item[0]])

        probable_parity_error_all.extend(parity_bit_indexes_incorrect)
        probable_parity_error_all = Counter(probable_parity_error_all).most_common()
        # Putting temporary weight on each parity indexes.
        # The default temporary weight will be number of occurrence of the parity bit from the data bit
        # The temporary weight will add 10 more if the specific parity bit didn't match in the first place.
        if false_positive:
            max_false_positive_in_parity = false_positive // 2
            if false_positive % 2 == 0:
                max_false_positive_in_data = false_positive // 2
            else:
                max_false_positive_in_data = false_positive // 2 + 1
        else:
            max_false_positive_in_parity = 0
            max_false_positive_in_data = 0

        # Initially matrix weight is 0
        matrix_weight = 0
        # Sum the probability of error for each cell for the parity bit.
        # Will also check how many false positive we are adding
        for item in probable_parity_error_all:
            temp_weight = item[1]
            if temp_weight >= threshold_parity:
                if matrix_copy[item[0][0]][item[0][1]] == 0:
                    probable_parity_error.append(item[0])
                elif max_false_positive_in_parity > false_positive_added_in_parity:
                    probable_parity_error.append(item[0])
                    false_positive_added_in_parity += 1
            matrix_weight += temp_weight
            # The parity that have more temporary weight have high probability of error.

        probable_error = []
        # Sum the probability of error for each cell for data bit
        for key in sorted(probable_data_error.keys(), reverse=True):
            if key >= threshold_data:
                for i in probable_data_error[key]:
                    if matrix_copy[i[0]][i[1]] == 0:
                        probable_error.append(i)
                    elif max_false_positive_in_data > total_false_positive_added:
                        probable_error.append(i)
                        total_false_positive_added += 1
            matrix_weight += key * len(probable_data_error[key])
        probable_error_data_parity = probable_error + probable_parity_error

        return matrix_copy, matrix_weight / len(parity_bit_indexes_correct), probable_error_data_parity

    def decode(self, data_stream, threshold_data, threshold_parity,
               maximum_number_of_error, false_positive):
        """
        Decode the given data stream into word and their respective index

        Parameters:
            data_stream: A string of 48 bit
            Otherwise only recovered word and position of error

        Return:
            decoded_data: A dictionary of index and world which is the most possible solution
            :param threshold_parity:
            :param data_stream:
            :param threshold_data:
            :param false_positive:
            :param maximum_number_of_error:
        """
        # If length of decoded data is not 48 then show error
        if len(data_stream) != 48:
            raise ValueError("The data stream length should be 48")
        # Initial check which parity bit index gave error and which gave correct results
        # Converting the data stream to data array first
        data_matrix_for_decoding = self.data_stream_to_matrix(data_stream)
        return self._decode(data_matrix_for_decoding, threshold_data,
                            threshold_parity, maximum_number_of_error, false_positive)

        #   After fixing orientation we need to check the checksum bit.
        #   If we check before orientation fixed then it will not work

        # sorting the matrix

    def check_checksum(self, matrix):
        for check_sum_bit in self.checksum_bit_relation:
            nearby_values = [int(matrix[a[0]][a[1]]) for a in self.checksum_bit_relation[check_sum_bit]]
            xor_value = reduce(lambda i, j: int(i) ^ int(j), nearby_values)
            if xor_value != matrix[check_sum_bit[0]][check_sum_bit[1]]:
                self.logger.info("Checksum did not matched")
                return False
        return True


# This is only for debugging purpose
if __name__ == "__main__":
    bin_stream = "0011011001010101"
    origami_object = Origami(2)
    encoded_file = origami_object.data_stream_to_matrix(origami_object.encode(bin_stream, 0, 16))

    encoded_file[1][0] = 0
    encoded_file[3][2] = 0
    encoded_file[0][6] = 0

    decoded_file = origami_object.decode(origami_object.matrix_to_data_stream(encoded_file), 2, 2, 9, 10)

    print(decoded_file)
    if not decoded_file == -1 and decoded_file['binary_data'] == bin_stream:
        print("Decoded successfully")
    else:
        print("wasn't decoded successfully")
