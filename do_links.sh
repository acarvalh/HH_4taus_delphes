#!/bin/tcsh
#find -type l -delete

ln -s ../Delphes_CMS/classes/SortableObject.h SortableObject.h
ln -s ../Delphes_CMS/classes/DelphesClasses.h DelphesClasses.h
ln -s ../Delphes_CMS/classes/                 classes
ln -s ../Delphes_CMS/external/ExRootAnalysis/ ExRootAnalysis
ln -s ../Delphes_CMS/libDelphes.so            libDelphes.so
ln -s ../Delphes_CMS/external/fastjet/        fastjet
ln -s ../Delphes_CMS/external                 external
