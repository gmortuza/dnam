# dNAM - digital Nucleic Acid Memory
This repository contains the encoding and decoding algorithm of dNAM.

## Requirements:
The codes are tested with **python 3.7**  
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy.
```bash
pip install numpy
```

## Usage
#### Encoding
User the following command to encode a given file to a list of origami matrices
```python
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
```python
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
sudo docker run -it dnam {dnam/encode.py}/{dnam/decode.py} [options]
```
For example to encode a file
```bash
sudo docker run -it dnam dnam/encode.py -f /dnam/test -o test_output.out
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
