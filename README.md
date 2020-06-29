# dNAM - digital Nucleic Acid Memory
This repository contains the error correction code and preprocessing code for dNAM

## Localization preprocessing analysis

### Requirements:
The codes are tested with **python 3.7**  
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages
```bash
pip install numpy scipy numba matplotlib lmfit tqdm yaml h5py

```
Or use the requirements.txt file:
```bash
pip install -r requirements.txt
```

(optional) To import PAINT/STORM movies (nd2, spe, tif, dax) directly into script, 3d_daostorm must be installed from
https://github.com/ZhuangLab/storm-analysis
Script can be run on localization data (csv, txt, hdf5) without 3d_daostorm installed. Acceptable formats include ThunderStorm, Picasso,
and 3d_daostorm.

### Usage of localization code
```
usage: dnam_mixed_origami_process.py [-h] [-f FILE] [-v] [-N NUMBER_CLUSTERS]
                                     [-s SKIP_CLUSTERS]
                                     [-d DRIFT_CORRECT_SIZE] [-ps PIXEL_SIZE]
                                     [-x XML] [-ft FILTER_FILE]
                                     [-gf GRID_FILE] [-gr GRID_SHAPE_ROWS]
                                     [-gc GRID_SHAPE_COLS]
                                     [-md MIN_DRIFT_CLUSTERS]
                                     [-gdx GLOBAL_DELTA_X]
                                     [-st SCALED_THRESHOLD] [-rf]
```

dNAM origami process script
```
optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  File name
  -v, --verbose         Print details of execution on console
  -N NUMBER_CLUSTERS, --number-clusters NUMBER_CLUSTERS
                        Number of clusters to analyze
  -s SKIP_CLUSTERS, --skip-clusters SKIP_CLUSTERS
                        Number of clusters to skip
  -d DRIFT_CORRECT_SIZE, --drift-correct-size DRIFT_CORRECT_SIZE
                        Final size of drift correct slice (0 is no drift
                        correction))
  -ps PIXEL_SIZE, --pixel-size PIXEL_SIZE
                        Pixel size. Needed if reading Thunderstorm csv
  -x XML, --xml XML     XML config file for 3d-daostorm analyis (default =
                        default_config.xml)
  -ft FILTER_FILE, --filter-file FILTER_FILE
                        XML filter file for post-drift correct filtering
                        (default is no filters)
  -gf GRID_FILE, --grid-file GRID_FILE
                        CSV file containing x,y coordinates of grid points
                        (default is DNAM average grid)
  -gr GRID_SHAPE_ROWS, --grid-shape-rows GRID_SHAPE_ROWS
                        Rows in the grid
  -gc GRID_SHAPE_COLS, --grid-shape-cols GRID_SHAPE_COLS
                        Columns in the grid
  -md MIN_DRIFT_CLUSTERS, --min-drift-clusters MIN_DRIFT_CLUSTERS
                        Min number of cluster to attempt fine drift correction
                        (default 5000)
  -gdx GLOBAL_DELTA_X, --global-delta-x GLOBAL_DELTA_X
                        Starting guess for global localization precision due
                        to drift correct, etc
  -st SCALED_THRESHOLD, --scaled-threshold SCALED_THRESHOLD
                        Threshold for binary counts, as a fraction of the
                        average of the 10 brightest points
  -rf, --redo-fitting   Redo grid fitting, even if fitted grid data exists
                      (has no effect on data without fits)
```

Example filter xml file:
```xml
<?xml version="1.0" encoding="iso-8859-1"?>
<filters>
  <!-- Valid filter names are frame, x, y, photons, sx, bg, and lpx. Every implemented filter must have
  "type" attribute as "absolute" or "percentile". "low" and "high" attributes must be set. Percentile
  filters must have values between 0.0 and 1.0. Low must be lower than high
  All units are pixels and photons
  Filters will be applied in listed order.-->
  <sx type="absolute" low="0.9" high="1.4"></sx>
  <photons type="percentile" low=".01" high ="0.95"></photons>
  <lpx type="percentile" low=".01" high="0.90"></lpx>
</filters>
```


## Error correction encoding/decoding algorithm

### Usage of error correction code
#### Encoding
User the following command to encode a given file to a list of origami matrices
```
python encode.py
                    -h , --help, show this help message and exit
                    -f , --file_in, file to encode
                    -o , --file_out, File to write the output
                    -r , --redundancy, Percentage of redundant origami
                    -fo, --formatted_output, Print the origami as matrix instead of single line
                    -v , --verbose, Print details on the console. 0 -> error. 1->debug, 2->info, 3->warning
                    -d , --degree, Degree old/new
```
#### Decoding
Use the following command to decode any encoded file:
```
python decode.py
                  -h, --help, show this help message and exit
                  -f , --file_in File to decode
                  -o , --file_out File to write output
                  -fz , --file_size, File size that will be decoded
                  -tp , --threshold_parity, Minimum weight for a parity bit cell to be consider that as an error
                  -td , --threshold_data, Minimum weight for a data bit cell to be consider as an error
                  -v , --verbose, Print details on the console. 0 -> error, 1 -> debug, 2 -> info, 3 -> warning
                  -r , --redundancy, How much redundancy was used during encoding
                  -ior, --individual_origami_info, Store individual origami information
                  -e , --error, Maximum number of error that the algorithm will try to fix
                  -fp , --false_positive, 0 can also be 1.
                  -d , --degree, Degree old/new
                  -cf , --correct_file, Original encoded file. Helps to check the status automatically.
```
#### Run using docker
You can use [docker](https://www.docker.com/) to run the algorithm  
To build the docker image use the following command
```bash
sudo docker build -t dnam .
```
Run the docker image as a container:
```bash
sudo docker run -it dnam {path/to/the/script/to_run} [options]
```
For example to encode a file
```bash
sudo docker run -it dnam error_correction/encode.py -f test.txt -o test_output.out
```
To copy the output file from docker container to host use:
```bash
sudo docker cp [container_name]:/[output_file_name] [path/to/copy/the/file]
```
For example:
```bash
sudo docker cp 98be599794ac:/test_output.out ./
```
To get the container_name use:
```bash
sudo docker ps -a
```
