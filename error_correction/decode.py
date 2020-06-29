import argparse
from processfile import ProcessFile


def read_args():
    """
    Read the arguments from command line
    :return:
    """
    parser = argparse.ArgumentParser(description="Decode a given origami matrices to a text file.")
    parser.add_argument("-f", "--file_in", help="File to decode", required=True)
    parser.add_argument("-o", "--file_out", help="File to write output", required=True)
    parser.add_argument("-fz", "--file_size", help="File size that will be decoded", type=int, required=True)
    parser.add_argument('-tp', '--threshold_parity',
                        help='Minimum weight for a parity bit cell to be consider that as an error', default=2, type=int)
    parser.add_argument("-td", "--threshold_data",
                        help='Minimum weight for a data bit cell to be consider as an error', default=2, type=int)
    parser.add_argument("-v", "--verbose", help="Print details on the console. "
                                                "0 -> error, 1 -> debug, 2 -> info, 3 -> warning", default=0, type=int)
    parser.add_argument("-r", "--redundancy", help="How much redundancy was used during encoding",
                        default=50, type=float)
    parser.add_argument("-ior", "--individual_origami_info", help="Store individual origami information",
                        action='store_true', default=True)
    parser.add_argument("-e", "--error", help="Maximum number of error that the algorithm "
                                              "will try to fix", type=int, default=8)
    parser.add_argument("-fp", "--false_positive", help="0 can also be 1.", type=int, default=0)

    parser.add_argument("-d", "--degree", help="Degree old/new", default="new", type=str)

    parser.add_argument("-cf", "--correct_file", help="Original encoded file. Helps to check the status automatically."
                        , type=str, default=False)

    args = parser.parse_args()
    return args


def main():
    args = read_args()
    dnam_decode = ProcessFile(redundancy=args.redundancy, verbose=args.verbose, degree=args.degree)
    dnam_decode.decode(args.file_in, args.file_out, args.file_size,
                       threshold_data=args.threshold_data,
                       threshold_parity=args.threshold_parity,
                       maximum_number_of_error=args.error,
                       false_positive=args.false_positive,
                       individual_origami_info=args.individual_origami_info,
                       correct_file=args.correct_file)


if __name__ == '__main__':
    main();
