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
print(inputFile)
file = open(os.getcwd()+'/samplesList.txt',"w")
file.write(str(glob.glob(str(inputFile)+'/*.root')))
file.close()
#######################  Output file
 
#orig_stdout=sys.stdout
#A=inputFile.split("/") 
#B=str(A[-1])
#C=B.split(".")
#print C[0]
#f=open(os.getcwd()+'/Folder_HHTo4T/'+'str(B[0])'+'_Out'+'.root',"w")    
#f = open(os.getcwd()+'/Folder_HHTo4T/'+str(C[0])+'_Out'+'.txt', 'w')
#sys.stdout = f



########################   matching
execfile("Matching_Taus.py")
execfile("Matching_Ws.py")
execfile("Matching_Zs.py")
#execfile("bjos_trying.py")
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
br_Hs=array('i',[0])
br_Es=array('i',[0])
br_Ms=array('i',[0])
# attach them to tree --  extend them to the number of categories
tree_name = "tree_22H"
tuple = ROOT.TTree(tree_name, tree_name)
tuple.Branch('nBs', br_nBs, 'nBs/I')
tuple.Branch('nJets', br_nJets, 'nJets/I')
tuple.Branch('Weights', br_Weights, 'Weights/D')
HH=tuple.Branch('Hs',br_Hs,'Hs/I')
E=tuple.Branch('Es',br_Es,'Es/I')
M=tuple.Branch('Ms',br_Ms,'Ms/I')

#############################################################
# Loop over file list
#############################################################
onlyCount = options.onlyCount
C=[]

totEvt = 0
nJets = 0
nBs = 0
Hs=0
Es=0
Ms=0
Weights = 1
Tau21 = -1.
N_Of_Ev_5tau=[]

histMass_T=ROOT.TH1D("Mass of Taus","Mass Of Taus",500,0,200)
histMass_W=ROOT.TH1D("Mass of the W-s","Mass of the W-s",1000,0,300)
histMass_Z=ROOT.TH1D("Mass of the Z-s","Mass of the Z-s",1000,0,300)
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
        for i in range(0,len(toProcess)):
            f=toProcess[i].split("/")
            C.append(f[-1])
        
    	#B=str(A[-1])
    	#C=B.split(".")
    	#print C[0]   
        execfile("bjos_trying.py")
        branchEvent = treeReader.UseBranch("Event")
        branchJet = treeReader.UseBranch("Jet")
        branchParticle = treeReader.UseBranch("Particle")
        for entry in range(0, numberOfEntries):
            #print "entry = "+str(entry)+" ================================================================="
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
            GenEs=[]
            GenMs=[]
            HH_TT=[]
            HH_WW=[]
            HH_ZZ=[]
            REAL_taus=[]           
            #N_Of_Ev_2tau=[]
            #s=0
            #########
           
            #print branchParticle.GetEntries()
            HiggsType = 0
            for part in range(0, branchParticle.GetEntries()):
                genparticle =  branchParticle.At(part)
                pdgCode = genparticle.PID
                #print ("pdgCode is: ",pdgCode)
                IsPU = genparticle.IsPU
                status = genparticle.M1
                charge=genparticle.Charge
               #########################
                if IsPU==0 and pdgCode==13 and genparticle.Status==1:                 
                    #print ("(pdgCode, genparticle.Status)____",pdgCode, genparticle.Status)
                    GenMs.append(genparticle)
                if IsPU==0 and pdgCode==11 and genparticle.Status==1:
                    #print ("(pdgCode, genparticle.Status)____",pdgCode, genparticle.Status)
                    GenEs.append(genparticle)
                if IsPU == 0 and pdgCode == 25 and genparticle.Status==22:
                    #print ("(pdgCode, genparticle.Status)____",pdgCode, genparticle.Status)
                    GenHs.append(genparticle)
                if IsPU == 0 and abs(pdgCode) == 24  and genparticle.Status == 22 :
                    #print (pdgCode, genparticle.Status)
                    GenVs.append(genparticle)
                    vectW=ROOT.TLorentzVector()
                    vectW.SetPtEtaPhiM(genparticle.PT,genparticle.Eta,genparticle.Phi,genparticle.Mass)
                    HH_WW.append(vectW)   
                if IsPU == 0 and abs(pdgCode) == 23  and genparticle.Status == 22 :
                    #print (pdgCode, genparticle.Status)
                    GenVs.append(genparticle)
                    vectZ=ROOT.TLorentzVector()
                    vectZ.SetPtEtaPhiM(genparticle.PT,genparticle.Eta,genparticle.Phi,genparticle.Mass)
                    HH_ZZ.append(vectZ)   
                if IsPU == 0 and (abs(pdgCode) == 15) and genparticle.Status == 2 :
                    #print ("(pdgCode, genparticle.Status)____",pdgCode, genparticle.Status)
                    GenTaus.append(genparticle)
                    vectT=ROOT.TLorentzVector()
                    vectT.SetPtEtaPhiM(genparticle.PT,genparticle.Eta,genparticle.Phi,genparticle.Mass)
                    HH_TT.append(vectT)
                    #print charge
                    #print ("in this event there are ",len(HH_TT) ,"particles")
                if IsPU == 0 and abs(pdgCode) == 5  and genparticle.Status == 23 :
                    #print (pdgCode, genparticle.Status)
                    if genparticle.PT > 10 :
                        dumb = ROOT.TLorentzVector()
                        dumb.SetPtEtaPhiM(genparticle.PT,genparticle.Eta,genparticle.Phi,genparticle.Mass)
                        GenBs.append(dumb)
            if len(HH_TT)==5 and len(GenHs)==2:
                N_Of_Ev_5tau.append(1)
            else:
                N_Of_Ev_5tau.append(0)
                
            #print ("number of Higgses / taus / V's / b's / E's / M's", len(GenHs), len(GenTaus), len(GenVs), len(GenBs), len(GenEs),len(GenMs))
            
            if len(GenHs) != 2 :
                print "not two H's"
                break
            #if len(GenVs)!=4 :
            #    print "Not 4 Bosons"
            #########################
   
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
                   ## using the DR with the genParticles to find out if there is a b-quark
            numbb = 0
            for i in range(0, len(RecoBJets)) : numbb += RecoBJets[i];
            nJets = len(RecoJets)
            nBs = numbb
            #######################
            br_nJets[0] = int(nJets)
            br_nBs[0] = int(nBs)
            tuple.Fill()
            
            ########################   Matching 2 Taus / 2V-s
            ifTaus=[]
            ifWs=[]
            ifZs=[]
            for i in range(0,branchParticle.GetEntries()):            
                HisMatchedTaus(HH_TT,ifTaus)
                HisMatchedWs(HH_WW,ifWs)
                HisMatchedZs(HH_ZZ,ifZs)
                
            ################  HIST.

            #print ("HH_TT-shi elementebis (anu tauebis) raodenobaa  ",len(HH_TT))
            
            for  i in range(0,len(HH_TT)):
                for j in range(i,len(HH_TT)):  
                    if i!=j:
                        n=(HH_TT[i]+HH_TT[j]).M()
      	                histMass_T.Fill(n)           
            for i in range(0,len(HH_WW)):
                for j in range(i,len(HH_WW)):
                    if i!=j:
                        n=(HH_WW[i]+HH_WW[j]).M()
                        histMass_W.Fill(n)
            for i in range(0,len(HH_ZZ)):
                for j in range(i,len(HH_ZZ)):
                    if i!=j:
                        n=(HH_ZZ[i]+HH_ZZ[j]).M()



            ######################## Filling Branches
    
            for i in range(0,branchParticle.GetEntries()):
                br_Hs[0]=int(len(GenHs))
                br_Es[0]=int(len(GenEs))
                br_Ms[0]=len(GenMs)
                HH.Fill()            
                E.Fill()
                M.Fill()
                
#########################
print "processed "+str(totEvt)+" "

if not onlyCount :
    for i in range(0,len(C)):
        out_file = ROOT.TFile(os.getcwd()+'/Folder_HHTo4T/'+str(C[i])+'_Out'+'.root', 'RECREATE')    ####    out_file = ROOT.TFile("teste111.root", 'RECREATE')
        out_file.WriteTObject(tuple, tuple.GetName(), 'Overwrite')
        out_file.Close()
#############################################################################################################################               
print("Number of events with 5 Taus  ",N_Of_Ev_5tau.count(1))
print ("Number of tau pairs", len(ifTaus))
print ("Number of  W pairs", len(ifWs))
print ("Number of Z pairs",len(ifZs))
###################### Hist. of masses of 2 Taus /// 2 Bozons

print ("Number of needed Tau pairs", ifTaus.count(1))
print ("Number of needed W pairs",ifWs.count(1))
print("Number of needed Z pairs",ifZs.count(1))
print("length of HH_TT / HH_WW // HH_ZZ ",len(HH_TT),len(HH_WW),len(HH_ZZ)) 

########################
c2=histMass_W.Draw()
c3=histMass_Z.Draw()
histMass_T.Draw()
#################################### FOLDER

#sys.stdout=orig_stdout
#f.close()



raw_input()




