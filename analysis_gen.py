#!/usr/bin/env python
import os, sys, time,math
import ROOT
import glob
from array import array
import numpy as np
pow = ROOT.TMath.Power
import bisect
from optparse import OptionParser
import matplotlib
import matplotlib.pyplot as plt
import pandas
#from root_pandas import read_root
import root_numpy
from root_numpy import root2array, rec2array, array2root, tree2array
execfile("functions.py")
# Delphes headers
ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gInterpreter.Declare('#include "DelphesClasses.h"')
ROOT.gSystem.Load("libDelphes")

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--input ", type="string", dest="input", help="A valid file in Delphes format. If you give somthing without .root extension it will interpret as a folder and try to process all the root files on that folder in series", default=" /eos/user/a/acarvalh/delphes_HH_4tau/HHTo4VTo4L/tree_HHTo4VTo4L_10.root")
parser.add_option("--onlyCount", action="store_true", dest="onlyCount", help="Only reports the number of events on the sample", default=False)
parser.add_option("--mass ", type="float", dest="mass", help="eventual sample-by sample cut", default=125.)
(options, args) = parser.parse_args()

inputFile = options.input
toProcess = [str(inputFile)]
if ".root" not in inputFile :
    toProcess = glob.glob(str(inputFile)+'/*.root')
print ("the first sample is: ", toProcess[0], " of a total of ",len(toProcess),"samples")

file = open(os.getcwd()+'/samplesList.txt',"w")
file.write(str(glob.glob(str(inputFile)+'/*.root')))
file.close()

#########################
# Cuts
#########################
execfile("cuts.py")
#########################
# Doing the trees with events
#########################
# declare branches
br_nBs = array('i', [0])
br_nJets = array('i', [0])
br_Weights  = array('d', [0.])
# attach them to tree --  extend them to the number of categories
tree_name = "tree_1"
tuple = ROOT.TTree(tree_name, tree_name)
tuple.Branch('nBs', br_nBs, 'nBs/I')
tuple.Branch('nJets', br_nJets, 'nJets/I')
tuple.Branch('Weights', br_Weights, 'Weights/D')
#############################################################
# Loop over file list
#############################################################
onlyCount = options.onlyCount
totEvt = 0
nJets = 0
nBs = 0
Weights = 1
Tau21 = -1.
for sample in toProcess :
    chain = ROOT.TChain("Delphes")
    #chain.Add(str(inputpath)+str(inputFile))
    try: chain.Add(sample)
    except IOError as e:
        print('Couldnt open the file (%s).' % e)
        continue
    # Create object of class ExRootTreeReader
    treeReader = ROOT.ExRootTreeReader(chain)
    numberOfEntries = treeReader.GetEntries()
    print sample+" has "+str(numberOfEntries)+" events "
    totEvt = totEvt + numberOfEntries
    if not onlyCount :
        #############################################################
        # Loop over all events
        #############################################################
        # Get pointers to branches used in this analysis
        branchEvent = treeReader.UseBranch("Event")
        branchJet = treeReader.UseBranch("Jet")
        branchJet = treeReader.UseBranch("JetPUPPI")
        branchParticle = treeReader.UseBranch("Particle")
        branchPhotonLoose = treeReader.UseBranch("PhotonLoose")
        branchPhotonTight = treeReader.UseBranch("PhotonTight")
        branchMuonLoose = treeReader.UseBranch("MuonLoose")
        branchMuonTight = treeReader.UseBranch("MuonTight")
        branchElectron = treeReader.UseBranch("Electron")
        branchMuonLooseCHS = treeReader.UseBranch("MuonLooseCHS")
        branchMuonTightCHS = treeReader.UseBranch("MuonTightCHS")
        branchElectronCHS = treeReader.UseBranch("ElectronCHS")
        branchMissingET = treeReader.UseBranch("MissingET")
        branchPuppiMissingET = treeReader.UseBranch("PuppiMissingET")
        branchScalarHT = treeReader.UseBranch("ScalarHT")

        for entry in range(0, numberOfEntries):
            print "entry = "+str(entry)+" ================================================================="
            # Load selected branches with data from specified event
            treeReader.ReadEntry(entry)
            ## NLO samples can have negative weights
            Weights = sign(branchEvent.At(0).Weight)
            #####################
            # Gen-level particles
            #####################
            GenBs = []
            GenVs = []
            GenTaus = []
            GenHs = []
            #print branchParticle.GetEntries()
            HiggsType = 0
            for part in range(0, branchParticle.GetEntries()):
               genparticle =  branchParticle.At(part)
               pdgCode = genparticle.PID
               #print pdgCode
               IsPU = genparticle.IsPU
               status = genparticle.M1
               #########################
               if IsPU == 0 and pdgCode == 25 and genparticle.Status == 22 :
                   # print (pdgCode, genparticle.Status)
                   GenHs.append(genparticle)
               if IsPU == 0 and abs(pdgCode) == 24  and genparticle.Status == 22 :
                  #print (pdgCode, genparticle.Status)
                  GenVs.append(genparticle)
               if IsPU == 0 and abs(pdgCode) == 23  and genparticle.Status == 22 :
                  #print (pdgCode, genparticle.Status)
                  GenVs.append(genparticle)
               if IsPU == 0 and (abs(pdgCode) == 15) and genparticle.Status == 2 :
                   #print (pdgCode, genparticle.Status)
                   GenTaus.append(genparticle)
               if IsPU == 0 and abs(pdgCode) == 5  and genparticle.Status == 23 :
                  #print (pdgCode, genparticle.Status)
                  if genparticle.PT > 10 :
                      dumb = ROOT.TLorentzVector()
                      dumb.SetPtEtaPhiM(genparticle.PT,genparticle.Eta,genparticle.Phi,genparticle.Mass)
                      GenBs.append(dumb)
            print ("number of Higgses / taus / V's / b's", len(GenHs), len(GenTaus), len(GenVs), len(GenBs))
            if len(GenHs) != 2 :
                print "not two H's"
                break
            #### Implement
            # findTauPairFromH
            # findTauPairFromV
            RecoJets = []
            RecoBJets = []
            for part in range(0, branchJet.GetEntries()): # add one more collection to the delphes card
                jet =  branchJet.At(part)
                if( jet.PT > 25 ) :
                   dumb = ROOT.TLorentzVector()
                   dumb.SetPtEtaPhiM(jet.PT,jet.Eta,jet.Phi,jet.Mass)
                   RecoJets.append(dumb)
                   RecoBJets.append(isbtagged(dumb, GenBs))
                   print jet.TauTag
                   ## using the DR with the genParticles to find out if there is a b-quark
            numbb = 0
            for i in range(0, len(RecoBJets)) : numbb += RecoBJets[i];
            nJets = len(RecoJets)
            nBs = numbb
            #######################
            br_nJets[0] = int(nJets)
            br_nBs[0] = int(nBs)
            tuple.Fill()
#########################
print "processed "+str(totEvt)+" "
if not onlyCount :
    out_file = ROOT.TFile("teste.root", 'RECREATE')
    out_file.WriteTObject(tuple, tuple.GetName(), 'Overwrite')
    out_file.Close()
