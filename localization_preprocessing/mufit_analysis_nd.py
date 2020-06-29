#!/usr/bin/env python
"""

Perform mufit analysis on a dax file given parameters.

Hazen 10/13
"""

import storm_analysis.daostorm_3d.find_peaks as find_peaks

import storm_analysis.sa_library.analysis_io as analysisIO
import storm_analysis.sa_library.parameters as params
import storm_analysis.sa_utilities.std_analysis as std_analysis
import storm_analysis.sa_library.datareader as datareader

import numpy
import os
from pims import ND2_Reader as nd2

class FrameReaderStdNd2(analysisIO.FrameReader):
    """
    Read frames from a 'standard' (as opposed to sCMOS) camera.

    Note: Gain is in units of ADU / photo-electrons.
    """
    def __init__(self,  movie_file = None, parameters = None, camera_gain = None, camera_offset = None, **kwds):
        #super(FrameReaderStd, self).__init__(**kwds)
        self.gain = None
        self.offset = None
        self.parameters = parameters
        self.rqe = 1.0
        self.verbose = 1
        if self.parameters is not None:
            self.verbose = (self.parameters.getAttr("verbosity") == 1)
        self.movie_data = Nd2Reader(movie_file)

        if camera_gain is None:
            self.gain = 1.0/self.parameters.getAttr("camera_gain")
            self.offset = self.parameters.getAttr("camera_offset")
        else:
            self.gain = 1.0/camera_gain
            self.offset = camera_offset
            
class Nd2Reader(datareader.Reader):
    """
    SPE (Roper Scientific) reader class.
    """
    
    def __init__(self, filename, verbose = False):
        super(Nd2Reader, self).__init__(filename, verbose = verbose)

        # open the file & read the header
        #self.header_size = 4100
        self.fileptr = nd2(filename)

        #self.fileptr.seek(42)
        frame_shape = self.fileptr.frame_shape
        self.image_width = int(frame_shape[0])
        #self.fileptr.seek(656)
        self.image_height = int(frame_shape[1])
        #self.fileptr.seek(1446)
        self.number_frames = len(self.fileptr)

 
        self.image_mode = self.fileptr.pixel_type
        if (self.image_mode == numpy.float32):
            self.image_size = 4 * self.image_width * self.image_height
        elif (self.image_mode == numpy.uint32):
            self.image_size = 4 * self.image_width * self.image_height
        elif (self.image_mode == numpy.int16):
            self.image_size = 2 * self.image_width * self.image_height
        elif (self.image_mode == numpy.uint16):
            self.image_size = 2 * self.image_width * self.image_height
        else:
            print("unrecognized spe image format: ", self.image_mode)

    def loadAFrame(self, frame_number, cast_to_int16 = True):
        """
        Load a frame & return it as a numpy array.
        """
        super(Nd2Reader, self).loadAFrame(frame_number)
        
        #self.fileptr.seek(self.header_size + frame_number * self.image_size)
        #image_data = numpy.fromfile(self.fileptr, dtype=self.image_mode, count = self.image_height * self.image_width)
        image_data = numpy.array(self.fileptr[frame_number].tolist())
        if cast_to_int16:
            image_data = image_data.astype(numpy.uint16)
        image_data = numpy.reshape(image_data, [self.image_height, self.image_width])
        return image_data


def analyze(movie_name, mlist_name, settings_name):

    # Load parameters.
    parameters = params.ParametersDAO().initFromFile(settings_name)

    # Check for possibly v1.0 parameters.
    if not parameters.hasAttr("background_sigma"):
        raise Exception("Parameter 'background_sigma' is missing. Version 1.0 parameters?")
    
    # Create finding and fitting object.
    finder = find_peaks.initFindAndFit(parameters)

    # Create object for reading (non sCMOS) camera frames.
    
    movie_ext = os.path.splitext(movie_name)[1]
    if movie_ext == '.nd2':
        frame_reader = FrameReaderStdNd2(movie_file = movie_name,
                                         parameters = parameters)
    else:
        frame_reader = analysisIO.FrameReaderStd(movie_file = movie_name,
                                             parameters = parameters)

    # Create movie reader (uses frame_reader).
    movie_reader = analysisIO.MovieReader(frame_reader = frame_reader,
                                          parameters = parameters)

    # Create localization file writer.
    data_writer = analysisIO.DataWriterHDF5(data_file = mlist_name,
                                            parameters = parameters,
                                            sa_type = '3D-DAOSTORM')

    # Run the analysis.
    std_analysis.standardAnalysis(finder,
                                  movie_reader,
                                  data_writer,
                                  parameters)


if (__name__ == "__main__"):

    import argparse

    parser = argparse.ArgumentParser(description = '3D-DAOSTORM analysis - Babcock, Optical Nanoscopy, 2012')

    parser.add_argument('--movie', dest='movie', type=str, required=True,
                        help = "The name of the movie to analyze, can be .dax, .tiff or .spe format.")
    parser.add_argument('--bin', dest='mlist', type=str, required=True,
                        help = "The name of the localizations output file. This is a binary file in HDF5 format.")
    parser.add_argument('--xml', dest='settings', type=str, required=True,
                        help = "The name of the settings xml file.")

    args = parser.parse_args()
    
    analyze(args.movie, args.mlist, args.settings)
    



#
# The MIT License
#
# Copyright (c) 2013 Zhuang Lab, Harvard University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
