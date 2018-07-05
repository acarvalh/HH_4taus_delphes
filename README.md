# HH_4taus_delphes

## Installation:

```
cmsrel CMSSW_8_1_0
cd CMSSW_8_1_0/src ; cmsenv (do this everytime you start a new shell)

git clone https://github.com/delphes/delphes.git Delphes_CMS
cd Delphes_CMS
git checkout tags/3.4.2pre15
./configure
sed -i -e 's/c++0x/c++1y/g' Makefile
make -j 4 (the first time make can break, try more than once)

cd ..
git clone https://github.com/acarvalh/HH_4taus_delphes.git
./do_links.sh (if you had installed delphes on another relative path adapt the paths in this file)
```

The signal samples are on /eos
An example for running the analysis is:

## Basic usage

```
./analysis_gen.py --input /eos/user/a/acarvalh/delphes_HH_4tau/HHTo4VTo4L/tree_HHTo4VTo4L_10.root
```
