import copy
import math
import os
import random
from collections import Counter
from itertools import chain
from operator import xor
import logging
import sys
from log import get_logger

sys.setrecursionlimit(100000)


class OrigamiPrePostProcess:
    """
    This class will handle the fountain code
    """

    def __init__(self, verbose):
        """
        Based on the file size the number of origami will be reduced automatically
        """
        self.number_of_bit_per_origami = 16  # This will update during creating the origami
        self.logger = get_logger(verbose=verbose, logger_name=__name__)

    def _get_droplet_details_old(self, total_origami_without_redundancy, total_origami):
        """
        This will return a degree distribution for our fountain code.
        This is the degree distribution that we used for our wetlab
        However this degree distribution isn't good enough for larger file
        So, later we developed another degree distribution that can handle both larger and smaller file
        :param total_origami_without_redundancy: Total number of origami that needed to be encoded
        :param total_origami: total_origami_without_redundancy + redundant_origami
        :return:
        """
        xored_map = []
        # The maximum degree won't be more than 15
        current_maximum_degree = current_degree = min(int(total_origami_without_redundancy * .7), 15)
        # We will start creating droplet from higher degree and will keep reducing
        current_degree_decreasing = True
        current_indexes = []
        droplet_generated_till_now = 0
        while len(xored_map) < total_origami:
            if len(xored_map) == 0:
                # All even
                even_indexes = list(range(0, total_origami_without_redundancy, 2))
                # split the even index by maximum number of droplet
                if len(even_indexes) <= current_maximum_degree:
                    xored_map.append(even_indexes)
                else:
                    start_from = 0
                    for end_till in range(current_maximum_degree, len(even_indexes) + current_maximum_degree,
                                          current_maximum_degree):
                        try:
                            xored_map.append(even_indexes[start_from:end_till])
                        except:
                            xored_map.append(even_indexes[start_from:])
                        start_from = end_till
                        droplet_generated_till_now += 1
                    # All odd
                odd_indexes = list(range(1, total_origami_without_redundancy, 2))
                start_from = 0
                if len(odd_indexes) <= current_maximum_degree:
                    xored_map.append(odd_indexes)
                else:
                    for end_till in range(current_maximum_degree, len(odd_indexes) + current_maximum_degree,
                                          current_maximum_degree):
                        try:
                            xored_map.append(odd_indexes[start_from:end_till])
                        except:
                            xored_map.append(odd_indexes[start_from:])
                        start_from = end_till
                        droplet_generated_till_now += 1
                continue

            if len(xored_map) % 2 == 0:
                # This is not the first droplet so we have to look for the previous droplet as well and remove first
                # two segments from previous droplet
                if len(current_indexes) == 0:
                    # This is the first droplet and this will be even
                    current_indexes = list(range(current_degree))
                else:
                    try:
                        start_index = current_indexes[2]
                    except IndexError:
                        start_index = (current_indexes[0] + (3 - len(current_indexes))) \
                                      % total_origami_without_redundancy
                    current_indexes = list(map(lambda x: x % total_origami_without_redundancy,
                                               list(range(start_index, start_index + current_degree))))
            else:
                if len(current_indexes) == 0:
                    # This is the first droplet and this will be even
                    current_indexes = list(range(current_degree))
                else:
                    try:
                        start_index = current_indexes[3]
                    except IndexError:
                        start_index = (current_indexes[0] + (
                                4 - len(current_indexes))) % total_origami_without_redundancy
                    current_indexes = list(map(lambda x: x % total_origami_without_redundancy,
                                               list(range(start_index, start_index + current_degree))))
            if current_indexes not in xored_map:
                xored_map.append(current_indexes)
            droplet_generated_till_now += 1
            if current_degree_decreasing:
                current_degree -= 1
            else:
                current_degree += 1
            if current_degree == 0:
                current_degree = 1
                current_degree_decreasing = False
            elif current_degree == current_maximum_degree:
                current_degree_decreasing = True
        return xored_map

    def _get_droplet_details(self, total_origami_without_redundancy, total_origami, initial_xor_map=[],
                             undecoded_elements=[]):
        """
        This will return a degree distribution for the droplet.
        This will work better with larger file
        First it will generate 'x' # of droplet. where x = # segment.
        After that it will analyze the previously created droplet and generate rest of the droplet which are redundant
        :param total_origami_without_redundancy:
        :param total_origami:
        :param initial_xor_map:
        :param undecoded_elements:
        :return:
        """
        xored_map = initial_xor_map
        current_maximum_degree = current_degree = min(int(total_origami_without_redundancy * .6),
                                                      15)  # Maximum degree will never be more than 15
        degree_increase = False
        droplet_generated = 0
        single_degree_droplet_so_far = []

        while len(xored_map) < total_origami_without_redundancy:
            # Generate two droplet here both reversed and forward direction
            forward_segment_starting_point = droplet_generated % total_origami_without_redundancy
            reversed_segment_starting_point = total_origami_without_redundancy - forward_segment_starting_point - 1
            single_droplet = [forward_segment_starting_point]
            while len(single_droplet) < current_degree:
                segment_to_add = (single_droplet[-1] + droplet_generated + 1) % total_origami_without_redundancy
                if segment_to_add in single_droplet:
                    segment_to_add = (segment_to_add + 1) % total_origami_without_redundancy
                single_droplet.append(segment_to_add)

            if set(single_droplet) not in xored_map:
                xored_map.append(set(single_droplet))
                if len(single_droplet) == 1:
                    single_degree_droplet_so_far.append(single_droplet)

            single_droplet = [reversed_segment_starting_point]
            while len(single_droplet) < current_degree:
                segment_to_add = (single_droplet[-1] - droplet_generated - 1) % total_origami_without_redundancy
                if segment_to_add in single_droplet:
                    segment_to_add = (segment_to_add - 1) % total_origami_without_redundancy
                single_droplet.append(segment_to_add)

            if set(single_droplet) not in xored_map:
                xored_map.append(set(single_droplet))
                if len(single_droplet) == 1:
                    single_degree_droplet_so_far.append(single_droplet)
                if current_degree == 0:
                    degree_increase = True
                elif current_degree == current_maximum_degree:
                    degree_increase = False
                if degree_increase:
                    current_degree += 1
                else:
                    current_degree -= 1
            droplet_generated += 1
        degree_generation_seed = 0
        # Every redundant droplet must give an unique single degree droplet

        while len(xored_map) < total_origami:
            # Get the droplet that has the minimum number of segment
            origami_need_to_generate = total_origami - len(xored_map)
            # try:
            newly_generated_droplets, single_degree_droplet_so_far = \
                self._find_next_redundant_droplet(xored_map, single_degree_droplet_so_far, origami_need_to_generate,
                                                  undecoded_elements)
            degree_generation_seed += 1
            for new_droplet in newly_generated_droplets:
                if len(xored_map) == total_origami:
                    break
                if new_droplet is not None and set(new_droplet) not in xored_map:
                    xored_map.append(set(new_droplet))
        return xored_map

    def _find_next_redundant_droplet(self, xored_map, single_degree_droplet_so_far, origami_need_to_generate,
                                     undecoded_elements):
        """
        This will analyze the previous set of droplet and will create new redundant droplet
        :param xored_map: Previous generated droplet
        :param single_degree_droplet_so_far:
        :param origami_need_to_generate: How many origami we need to generate
        :param undecoded_elements: Some of the droplet cannot be decoded. List of them. So that new generated droplet can address them
        :return:
        """
        flat_degree = list(chain.from_iterable(xored_map))
        segment_appear = Counter(flat_degree)
        random.seed(len(xored_map))
        shuffled_xored_map = copy.deepcopy(xored_map)
        random.shuffle(shuffled_xored_map)
        segment_appear = sorted(segment_appear.items(), key=lambda item: item[1])
        newly_generated_droplet = []
        origami_with_less_appear_segment = int(origami_need_to_generate * .1)
        number_of_less_appeared_droplet = min(int(len(segment_appear) * .3), origami_with_less_appear_segment)
        less_appeared_segment = []
        for single_appear_droplet in segment_appear:
            less_appeared_segment.append(single_appear_droplet[0])
            if number_of_less_appeared_droplet <= len(less_appeared_segment):
                break
        # Generate_droplet with less appeared segment
        for i in range(origami_with_less_appear_segment):
            new_droplet = set(random.sample(less_appeared_segment,
                                            random.choice(list(range(1, min(10, len(less_appeared_segment)) + 1)))))
            if new_droplet not in xored_map:
                newly_generated_droplet.append(new_droplet)

        if not undecoded_elements:
            # We don't have the undecoded droplet at this point so we will make the
            # droplet so that most appeared droplet becomes free
            # As calculating this process is computationally expensive we will generate total 10 droplet at a time
            newly_generated_droplet, single_degree_droplet_so_far = self._generate_droplet_analyzing_generated_droplet(
                newly_generated_droplet, origami_need_to_generate,
                segment_appear, single_degree_droplet_so_far,
                shuffled_xored_map, xored_map)
        else:
            # We have undecoded elements so we can analyze that
            # free first 3 element that appear most in degree 3
            all_two_degree_droplet_flatten = []
            all_three_degree_droplet_flatten = []
            all_three_degree_droplet = []
            all_two_degree_droplet = []
            freed = []
            # Seperating 3 & 2 degree droplet
            undecoded_elements.append([])
            for index in range(len(undecoded_elements)):
                if len(undecoded_elements[index]) == 2:
                    all_two_degree_droplet_flatten.extend(list(undecoded_elements[index]))
                    all_two_degree_droplet.append(undecoded_elements[index])
                elif len(undecoded_elements[index]) == 3:
                    all_three_degree_droplet_flatten.extend(list(undecoded_elements[index]))
                    all_three_degree_droplet.append(undecoded_elements[index])
                else:
                    del undecoded_elements[-1]
                    undecoded_elements = undecoded_elements[index:]
                    break

            picking_from_two_degree_droplet = 0
            generated_from_two = 0
            most_common_from_two = Counter(all_two_degree_droplet_flatten).most_common()
            for most_common_element_in_two_degree_droplet in most_common_from_two:
                if len(newly_generated_droplet) >= int(origami_need_to_generate * .6) or generated_from_two >= len(
                        all_two_degree_droplet):
                    break
                if most_common_element_in_two_degree_droplet[0] not in freed:
                    for i in range(picking_from_two_degree_droplet, len(all_two_degree_droplet)):
                        picking_from_two_degree_droplet += 1
                        new_droplet = set(all_two_degree_droplet[i])
                        freed.extend(list(new_droplet))
                        if most_common_element_in_two_degree_droplet[0] in new_droplet:
                            new_droplet.remove(most_common_element_in_two_degree_droplet[0])
                        else:
                            new_droplet.add(most_common_element_in_two_degree_droplet[0])
                        if new_droplet not in newly_generated_droplet and new_droplet not in xored_map \
                                and not len(new_droplet) == 0:
                            newly_generated_droplet.append(new_droplet)
                            generated_from_two += 1
                            break

            picking_from_x_degree_droplet = 0
            generated_from_three = 0
            most_common_from_three = Counter(all_three_degree_droplet_flatten).most_common()
            for most_common_in_here in most_common_from_three:
                if len(newly_generated_droplet) >= int(
                        origami_need_to_generate * .2) or generated_from_three >= generated_from_three:
                    break
                if most_common_in_here[0] not in freed:
                    for i in range(picking_from_x_degree_droplet, len(all_three_degree_droplet)):
                        picking_from_x_degree_droplet += 1
                        new_droplet = set(all_three_degree_droplet[i])
                        freed.append(most_common_in_here[0])
                        if most_common_in_here[0] in new_droplet:
                            new_droplet.remove(most_common_in_here[0])
                        else:
                            new_droplet.add(most_common_in_here[0])

                        if new_droplet not in newly_generated_droplet and new_droplet not in xored_map \
                                and not len(new_droplet) == 0:
                            newly_generated_droplet.append(new_droplet)
                            generated_from_three += 1
                            break

            flat_degree_rest_unrecoverable = list(chain.from_iterable(undecoded_elements))
            flat_degree_rest_unrecoverable_most_common = Counter(flat_degree_rest_unrecoverable).most_common()
            picking_from_x_degree_droplet = 0
            for most_common_in_here in flat_degree_rest_unrecoverable_most_common:
                if len(newly_generated_droplet) >= origami_need_to_generate:
                    break
                if most_common_in_here[0] not in freed:
                    for i in range(picking_from_x_degree_droplet, len(undecoded_elements)):
                        picking_from_x_degree_droplet += 1
                        new_droplet = set(undecoded_elements[i])
                        if most_common_in_here[0] in new_droplet:
                            new_droplet.remove(most_common_in_here[0])
                        else:
                            new_droplet.add(most_common_in_here[0])
                        if new_droplet not in newly_generated_droplet and new_droplet not in xored_map \
                                and not len(new_droplet) == 0:
                            newly_generated_droplet.append(new_droplet)
                            freed.append(most_common_in_here[0])
                            break

            newly_generated_droplet, single_degree_droplet_so_far = self._generate_droplet_analyzing_generated_droplet(
                newly_generated_droplet, origami_need_to_generate,
                segment_appear, single_degree_droplet_so_far,
                shuffled_xored_map, xored_map)

        return newly_generated_droplet, single_degree_droplet_so_far

    def _generate_more_droplet(self, data_map, single_element, xored_element_done, total_origami_without_redundancy,
                               generate_droplet):
        """
        This is used to decode the fountain code.
        This will generate newer droplet based on the subset decoding
        If a droplet is a subset of another droplet then we will XOR them
        and will get a newer droplet with less degree
        :param data_map: Degree distribution
        :param single_element: Single degree droplet
        :param xored_element_done: Droplet that we have already tried XORing
        :param total_origami_without_redundancy: Number of file segment
        :param generate_droplet:
        :return:
        """
        # Generate more two degree droplet
        all_elements = sorted(set(data_map), key=len)
        # Tuple length only generate_droplet
        elements = []
        for element in all_elements:
            if len(element) <= generate_droplet:
                elements.append(element)
            elif len(element) > generate_droplet:
                break
        for starting_index, element_1 in enumerate(elements):  # XOR 1
            xor_element_1, xor_data_1 = element_1, data_map[element_1]
            for element_2 in elements[starting_index + 1:]:
                xor_element_2, xor_data_2 = element_2, data_map[element_2]
                # Now we will perform xor and look for new `generate_droplet`
                after_xor_element = tuple(
                    sorted(set(xor_element_1).union(set(xor_element_2)) - set(xor_element_1).intersection(
                        set(xor_element_2))))
                if min(len(xor_element_1), len(xor_element_2)) >= len(after_xor_element) > 0:
                    data_map[tuple(sorted(after_xor_element))] = format(xor(int(xor_data_1, 2), int(xor_data_2, 2)),
                                                                        '0' + str(self.number_of_bit_per_origami) + 'b')
        return self._process_xoring(data_map, single_element, xored_element_done, total_origami_without_redundancy,
                                    generate_droplet)

    def _process_xoring(self, data_map, single_element, xored_element_done, total_origami_without_redundancy,
                        generate_droplet):
        """

        :param data_map:
        :param single_element:
        :param xored_element_done:
        :param total_origami_without_redundancy:
        :param generate_droplet:
        :return:
        """
        elements = sorted(data_map, key=len)
        if total_origami_without_redundancy == len(single_element):
            return single_element, elements, data_map  # Recovered successfully
        if set(elements).issubset(xored_element_done):
            # Try to generate more 2 degree droplet
            if len(elements) == 0 or len(elements[-1]) < generate_droplet:
                return single_element, elements, data_map  # wasn't able to recover
            return self._generate_more_droplet(data_map, single_element, xored_element_done,
                                                   total_origami_without_redundancy, generate_droplet + 1)
            # return single_element  # was not able recover
        # Xor first element with all other elements and put that element as done xoring
        for element in elements:
            xor_element, xor_data = element, data_map[element]
            if xor_element not in xored_element_done:
                break
        for element in data_map.copy():
            if set(xor_element).issubset(set(element)) and set(element).issubset(set(xor_element)):
                continue  # Don't xor with itself
            if set(xor_element).issubset(set(element)):
                data_map[tuple(sorted(set(element) - set(xor_element)))] = format(
                    xor(int(xor_data, 2), int(data_map[element], 2)),
                    '0' + str(self.number_of_bit_per_origami) + 'b')
                del data_map[element]
        if len(xor_element) == 1:
            single_element[xor_element[0]] = xor_data
            del data_map[tuple(sorted(xor_element))]
        else:
            xored_element_done.append(xor_element)

        return self._process_xoring(data_map, single_element, xored_element_done, total_origami_without_redundancy,
                                    generate_droplet)

    def _generate_droplet_analyzing_generated_droplet(self, newly_generated_droplet, origami_need_to_generate,
                                                      segment_appear, single_degree_droplet_so_far,
                                                      shuffled_xored_map, xored_map):
        random_shuffle_index = 0
        while len(newly_generated_droplet) < origami_need_to_generate:
            if not segment_appear:
                break
            most_appeared_segment = segment_appear[-1][0]
            del segment_appear[-1]
            if [most_appeared_segment] not in single_degree_droplet_so_far:
                for single_droplet_index in range(random_shuffle_index, len(shuffled_xored_map)):
                    random_shuffle_index += 1
                    if most_appeared_segment in shuffled_xored_map[single_droplet_index]:
                        shuffled_xored_map[single_droplet_index].remove(most_appeared_segment)
                    else:
                        shuffled_xored_map[single_droplet_index].add(most_appeared_segment)
                    single_degree_droplet_so_far.append([most_appeared_segment])
                    if shuffled_xored_map[single_droplet_index] not in newly_generated_droplet \
                            and shuffled_xored_map[single_droplet_index] not in xored_map \
                            and not len(shuffled_xored_map[single_droplet_index]) == 0:
                        newly_generated_droplet.append(shuffled_xored_map[single_droplet_index])
                        break

        return newly_generated_droplet, single_degree_droplet_so_far

    def _generate_random_data_map(self, degree_distribution, segment, red, initial_xor_map=[], undecoded_elements=[]):
        """
        This is used for simulation only
        :param degree_distribution:
        :param segment:
        :param red:
        :param initial_xor_map:
        :param undecoded_elements:
        :return:
        """
        if degree_distribution == "old":
            xored_map = self._get_droplet_details_old(segment, math.ceil(segment * red))
        else:
            xored_map = self._get_droplet_details(segment, math.ceil(segment * red), initial_xor_map,
                                                  undecoded_elements)

        data_map = {}
        for i in range(len(xored_map)):
            try:
                data_map[tuple(sorted(xored_map[i]))] = format(i + 1, '0' + str(self.number_of_bit_per_origami) + 'b')
            except KeyError as e:
                logging.err("Missing origami: {i}".format(i=i))
        return data_map, xored_map

    def recoverable_red(self, total_bits, min_required_redundancy, degree, last_bit_stored_per_origami=16):
        """
        This is used for simulation only
        :param total_bits:
        :param min_required_redundancy:
        :param degree:
        :param last_bit_stored_per_origami:
        :return:
        """
        for bit_store_per_origami in reversed(range(1, last_bit_stored_per_origami + 1)):
            initial_xor_map = []
            undecoded_elements = []
            # Newer degree distribution will look for previous generated droplet to
            # create new droplet. So, instead of creating all redundant droplet we will start from
            # redundancy 0
            if degree == "new":
                redundancy = 0
            else:
                redundancy = 50
            #  Checking if that bit works or not
            # we want to make sure that every file is decodable. So we will start not use fixed redundancy. We will
            # start our redundancy level from 50%. an increase by two. to check if now that's in recoverable state.
            while True:
                total_origami_needed = math.ceil(total_bits / bit_store_per_origami)
                total_origami_with_red = math.ceil(total_origami_needed * ((100 + redundancy) / 100))
                bits_for_index_remained = 20 - bit_store_per_origami
                if total_origami_with_red <= 2 ** bits_for_index_remained:
                    segments = math.ceil(total_bits / bit_store_per_origami)
                    return_segment_from_simulation, initial_xor_map, undecoded_elements = \
                        self._simulate_degree_distribution(segments, ((100 + redundancy) / 100),
                                                           initial_xor_map, degree, undecoded_elements)
                    if return_segment_from_simulation == segments and redundancy >= min_required_redundancy:
                        self.number_of_bit_per_origami = bit_store_per_origami
                        return segments, total_origami_with_red, redundancy, bit_store_per_origami, initial_xor_map
                    else:
                        # we weren't able to recover the file with our previous
                        # redundancy so we are increasing redundancy
                        redundancy += 10
                else:
                    break  # it will decrease the bit per origami

    def _simulate_degree_distribution(self, segment, red, initial_xor_map=[], degree="new", undecoded_elements=[]):
        """
        This is used to check if the current droplet can recover the original file or not
        :param segment:
        :param red:
        :param initial_xor_map:
        :param degree:
        :param undecoded_elements:
        :return:
        """
        data_map, xor_map = self._generate_random_data_map(degree, segment, red, initial_xor_map, undecoded_elements)
        single_element = {}
        xored_element_done = []
        single_element, undecoded_elements, _ = self._process_xoring(data_map,
                                                                     single_element, xored_element_done, segment, 1)
        return len(single_element), xor_map, undecoded_elements

    def decode(self, data_dictionary, segment, xored_map, single_element, data_map):
        """
        Decode the fountain code decoding
        :param data_dictionary:
        :param segment:
        :param xored_map:
        :param single_element:
        :param data_map:
        :return:
        """
        for i in data_dictionary.keys():
            try:
                # a element in the xored_map is already decoded we will check that. And xor that in here
                segments_in_degree = xored_map[i]
                droplet_data = data_dictionary[i]
                for single_segment_in_degree in segments_in_degree.copy():
                    if single_segment_in_degree in single_element:
                        droplet_data = format(
                            xor(int(droplet_data, 2), int(single_element[single_segment_in_degree], 2)),
                            '0' + str(self.number_of_bit_per_origami) + 'b')
                        segments_in_degree.remove(single_segment_in_degree)
                if len(segments_in_degree) > 0:
                    data_map[tuple(segments_in_degree)] = droplet_data
            except KeyError as e:
                # from the majority voting queue we will add more origamies in each steps
                # So we don't need to care about missing origami's now
                # The missing origami is calculated before calling this function
                pass
        xored_element_done = []
        decoded_elements, undecoded_elements, current_data_map = self._process_xoring(data_map, single_element,
                                                                                      xored_element_done, segment, 1)
        return decoded_elements, current_data_map

    def encode(self, binary_stream, min_required_redundancy=.1, degree="new"):
        """
        Generate the droplet
        :param binary_stream: Binary of the file that will be encoded
        :param min_required_redundancy:
        :param degree: Which degree distribution we will use
        :return:
        """
        # Adding the padding
        segments, _, redundancy, self.number_of_bit_per_origami, xored_map = \
            self.recoverable_red(len(binary_stream), min_required_redundancy, degree)
        while len(binary_stream) % self.number_of_bit_per_origami != 0:
            binary_stream += '0'
        data_list = [binary_stream[
                     i * self.number_of_bit_per_origami:i * self.number_of_bit_per_origami
                                                        + self.number_of_bit_per_origami]
                     for i in range(segments)]  # Divide the data
        xored_data = []
        for single_xor_map in xored_map:
            single_xor_map = list(single_xor_map)
            single_xor_data = int(data_list[single_xor_map[0]], 2)
            for i, index in enumerate(single_xor_map):
                if i == 0:
                    continue
                single_xor_data = xor(int(single_xor_data), int(data_list[index], 2))
            xored_data.append(format(single_xor_data, '0' + str(self.number_of_bit_per_origami) + 'b'))

        self.logger.info(str(len(xored_data)) + " droplet created for " + str(segments) + " segments")
        return segments, xored_data, self.number_of_bit_per_origami, redundancy


if __name__ == '__main__':

    def process_simulation(start, end, old_new):
        last_bit_stored_per_origami = 16
        file_interval = 512  # Interval of file checking

        if old_new == "old":
            file_name = "../test/test_result/old_deg_start_" + str(start) + "_end_" + str(end) + ".csv"
        else:
            file_name = "../test/test_result/new_deg_start_" + str(start) + "_end_" + str(end) + ".csv"

        if os.path.exists(file_name):
            file = open(file_name, "a")
        else:
            file = open(file_name, "w")
            file.write("File Size(bytes),without redundancy, with redundancy, redundancy\n")

        for file_size in range(start, end, file_interval):
            preprocess_post_process = OrigamiPrePostProcess(verbose=1)
            without_red, with_red, redundancy, last_bit_stored_per_origami, _ = preprocess_post_process.recoverable_red(
                file_size * 8, 0, old_new, last_bit_stored_per_origami)
            file.write("{file_size},{without_red},{with_red},{redundancy}\n".format(file_size=file_size,
                                                                                    without_red=without_red,
                                                                                    with_red=with_red,
                                                                                    redundancy=redundancy))
            file.flush()

        file.close()


    old_new = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])

    process_simulation(start, end, old_new)
