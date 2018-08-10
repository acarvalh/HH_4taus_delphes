#!/usr/bin/env python
import os, sys, time,math
import ROOT
from ROOT import TCanvas
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
#print(inputFile)
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
########################### TREE OF CATEGORIES
a8l_0t=[]                                       ##### numb of event with this category
a3l_1t=[]                                       ##### numb of event with this category
a2l_2t=[]                                       ##### numb of event with this category
a1l_3t=[]                                       ##### numb of event with this category
a01l_0t=[]                                      ##### numb of event with this category
a2l_0t=[]                                       ##### numb of event with this category
a3l_0t=[]                                       ##### numb of event with this category
a2l_1t=[]                                       ##### numb of event with this category
a1l_2t=[]                                       ##### numb of event with this category
a5l_1t=[]                                       ##### numb of event with this category
a4l_2t=[]                                       ##### numb of event with this category
a0l_2t=[]                                       ##### numb of event with this category
jet_in_8l_0t=[]
jet_in_3l_1t=[]
jet_in_2l_2t=[]
jet_in_1l_3t=[]
jet_in_01l_0t=[]
jet_in_2l_0t=[]
jet_in_3l_0t=[]
jet_in_2l_1t=[]
jet_in_1l_2t=[]
jet_in_5l_1t=[]
jet_in_4l_2t=[]
jet_in_0l_2t=[]

fi=ROOT.TFile("Categories.root","recreate")
A=['l>=4_0tauh','3l_1tauh','2l_2tauh','1l_3tauh','l<=1_0tauh','2l_0tauh','3l_0tauh','2l_1tauh','1l_2tauh','5l_1tauh','4l_2tauh','0l_2tauh']
print("number of categories___",len(A))
br_Njets=array('i',[0])
br_countEvent=array('i',[0])

a=ROOT.TTree("l>=4_0tauh","tree title")
b=ROOT.TTree('3l_1tauh',"tree title")
c=ROOT.TTree('2l_2tauh',"tree title") 
d=ROOT.TTree('1l_3tauh',"tree title")
e=ROOT.TTree('l<=1_0tauh',"tree title")
m=ROOT.TTree('2l_0tauh',"tree title")
p=ROOT.TTree('3l_0tauh',"tree title")
h=ROOT.TTree('2l_1tauh',"tree title")
ii=ROOT.TTree('1l_2tauh',"tree title")
jj=ROOT.TTree('5l_1tauh',"tree title")
k=ROOT.TTree('4l_2tauh',"tree title")
l=ROOT.TTree('0l_2tauh',"tree title")

a.Branch('countEvent',br_countEvent,'countEvent/I')
b.Branch('countEvent',br_countEvent,'countEvent/I')
c.Branch('countEvent',br_countEvent,'countEvent/I')
d.Branch('countEvent',br_countEvent,'countEvent/I')
e.Branch('countEvent',br_countEvent,'countEvent/I')
m.Branch('countEvent',br_countEvent,'countEvent/I')
p.Branch('countEvent',br_countEvent,'countEvent/I')
h.Branch('countEvent',br_countEvent,'countEvent/I')
ii.Branch('countEvent',br_countEvent,'countEvent/I')
jj.Branch('countEvent',br_countEvent,'countEvent/I')
k.Branch('countEvent',br_countEvent,'countEvent/I')
l.Branch('countEvent',br_countEvent,'countEvent/I')

a.Branch('Njets',br_Njets,'Njets/I')
b.Branch('Njets',br_Njets,'Njets/I')
c.Branch('Njets',br_Njets,'Njets/I')
d.Branch('Njets',br_Njets,'Njets/I')
e.Branch('Njets',br_Njets,'Njets/I')
m.Branch('Njets',br_Njets,'Njets/I')
p.Branch('Njets',br_Njets,'Njets/I')
h.Branch('Njets',br_Njets,'Njets/I')
ii.Branch('Njets',br_Njets,'Njets/I')
jj.Branch('Njets',br_Njets,'Njets/I')
k.Branch('Njets',br_Njets,'Njets/I')
l.Branch('Njets',br_Njets,'Njets/I')
#print c
fi.Write()
print c
########################   matching
execfile("Matching_Taus.py")
execfile("Matching_Ws.py")
execfile("Matching_Zs.py")
execfile("func_of_genp.py")
#execfile("Treeee.py")
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

histNumb_of_jets_8l_0t=ROOT.TH1D("number of jets in category l>=4_0tauh","number of jets in category l>=4_0tauh",20,0,20)
histNumb_of_jets_3l_1t=ROOT.TH1D("number of jets in category 3l_1tauh","number of jets in category 3l_1tauh",20,0,20)
histNumb_of_jets_2l_2t=ROOT.TH1D("number of jets in category 2l_2tauh","number of jets in category 2l_2tauh",20,0,20)
histNumb_of_jets_1l_3t=ROOT.TH1D("number of jets in category 1l_3tauh","number of jets in category 1l_3tauh",20,0,20)
histNumb_of_jets_01l_0t=ROOT.TH1D("number of jets in category l<=1_0tauh","number of jets in category l<=1_0tauh",20,0,20)
histNumb_of_jets_2l_0t=ROOT.TH1D("number of jets in category 2l_0tauh","number of jets in category 2l_0tauh",20,0,20)
histNumb_of_jets_3l_0t=ROOT.TH1D("number of jets in category 3l_0tauh","number of jets in category 3l_0tauh",20,0,20)
histNumb_of_jets_2l_1t=ROOT.TH1D("number of jets in category 2l_1tauh","number of jets in category 2l_1tauh",20,0,20)
histNumb_of_jets_1l_2t=ROOT.TH1D("number of jets in category 1l_2tauh","number of jets in category 1l_2tauh",20,0,20)
histNumb_of_jets_5l_1t=ROOT.TH1D("number of jets in category 5l_1tauh","number of jets in category 5l_1tauh",20,0,20)
histNumb_of_jets_4l_2t=ROOT.TH1D("number of jets in category 4l_2tauh","number of jets in category 4l_2tauh",20,0,20)
histNumb_of_jets_0l_2t=ROOT.TH1D("number of jets in category 0l_2tauh","number of jets in category 0l_2tauh",20,0,20)


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
        branchEvent = treeReader.UseBranch("Event")
        branchJet = treeReader.UseBranch("Jet")
        branchJet = treeReader.UseBranch("JetPUPPI")
        branchParticle = treeReader.UseBranch("Particle")
        branchPhotonLoose = treeReader.UseBranch("PhotonLoose")
        branchPhotonTight = treeReader.UseBranch("PhotonTight")
        branchMuonLoose = treeReader.UseBranch("MuonLoose")  #
        branchMuonTight = treeReader.UseBranch("MuonTight")
        branchElectron = treeReader.UseBranch("Electron")    #
        branchMuonLooseCHS = treeReader.UseBranch("MuonLooseCHS")
        branchMuonTightCHS = treeReader.UseBranch("MuonTightCHS")
        branchElectronCHS = treeReader.UseBranch("ElectronCHS")
        branchMissingET = treeReader.UseBranch("MissingET")
        branchPuppiMissingET = treeReader.UseBranch("PuppiMissingET")
        branchScalarHT = treeReader.UseBranch("ScalarHT")

        for i in range(0,len(toProcess)):
            f=toProcess[i].split("/")
            C.append(f[-1])
        ##execfile("bjos_trying.py")
    	##B=str(A[-1])
    	##C=B.split(".")
    	##print C[0]   
        #branchEvent = treeReader.UseBranch("Event")
        #branchJet = treeReader.UseBranch("Jet")
        #branchParticle = treeReader.UseBranch("Particle")
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
            vectZ=ROOT.TLorentzVector()
            vectW=ROOT.TLorentzVector()
            vectT=ROOT.TLorentzVector()
            dumb=ROOT.TLorentzVector()          
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
                g(GenBs,GenVs,GenTaus,GenHs,GenEs,GenMs,HH_TT,HH_WW,HH_ZZ,vectZ,vectT,vectW,dumb)
                #print ("number of Higgses / taus / V's / b's / E's / M's", len(GenHs), len(GenTaus), len(GenVs), len(GenBs), len(GenEs),len(GenMs)) 
            #########################
            if len(HH_TT)==5 and len(GenHs)==2:
                N_Of_Ev_5tau.append(1)
            else:
                N_Of_Ev_5tau.append(0)
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
            ######################## CATEGORIES
            muon_charge=[]
            electron_charge=[]
            tauhs_charge=[]
            Muons=[]  ###### if yes, 1...if no 0
            Els=[]    ###### if yes, 1...if no 0
            tauhs=[]  ###### if yes, 1...if no 0
            JET=[]    ###### other than tauhs
          
            #print c
            ############################################################  M U O N
            for part in range(0,branchMuonLoose.GetEntries()):
                muon=branchMuonLoose.At(part)
                if muon.PT>9 and abs(muon.Eta)<2.4:
                    vectmuon=ROOT.TLorentzVector(muon.PT,muon.Eta,muon.Phi,muon.T)
                    #vectmuon.SetPtEtaPhiT(muon.PT,muon.Eta,muon.Phi,muon.T) 
                    Muons.append(vectmuon)
                    muon_charge.append(muon.Charge)
                
            numbMuons=np.sum(len(Muons))  ######## those events that contains those 2 muons
            
            ############################################################# E L E C T R O N
            for part in range(0,branchElectron.GetEntries()):
                electron=branchElectron.At(part)
                if electron.PT>13 and abs(electron.Eta)<2.5: 
                    vectelect=ROOT.TLorentzVector(electron.PT,electron.Eta,electron.Phi,electron.T)
                    #vectelect.SetPtEtaPhiT(muon.PT,muon.Eta,muon.Phi,muon.T) 
                    Els.append(vectelect)
                    electron_charge.append(electron.Charge)
                
	    numbEls=np.sum(len(Els))   ######## those events that contains those 2 electrons
            

            ############################################################## JETS/TAUHS

            for part in range(0, branchJet.GetEntries()): # add one more collection to the delphes card
                jet =  branchJet.At(part)
                if jet.TauTag:
                    if jet.PT > 20 and abs(jet.Eta)<2.3 :
                        vecttau=ROOT.TLorentzVector()
                        vecttau.SetPtEtaPhiM(jet.PT,jet.Eta,jet.Phi,jet.Mass)
                        tauhs.append(vecttau)
                        tauhs_charge.append(jet.Charge)
                else: JET.append(1)
            
	    numb_with_tauhs=np.sum(len(tauhs))    ######## those events that contains those 2 tauhs
            
            
            ##################### counting events in Categories

            if numbMuons+numbEls>=4 and numb_with_tauhs==0:
                a8l_0t.append(1)
                jet_in_8l_0t.append(JET.count(1))
            else:
                a8l_0t.append(0)
                jet_in_8l_0t.append(0)
            ####
    	    if numbMuons+numbEls==3 and numb_with_tauhs==1:
		a3l_1t.append(1)
		jet_in_3l_1t.append(JET.count(1))
	    else:
		a3l_1t.append(0)
		jet_in_3l_1t.append(0)
            ####
            if numbMuons==2 and numb_with_tauhs==2 or numbEls==2 and numb_with_tauhs==2 or numbEls==1 and numbMuons==1 and numb_with_tauhs==2:
	    #if numbMuons+numbEls==2 and numb_with_tauhs==2:
                a2l_2t.append(1)
                jet_in_2l_2t.append(JET.count(1))
            else: 
		a2l_2t.append(0)
		jet_in_2l_2t.append(0)
            ####
            if numbMuons+numbEls==1 and numb_with_tauhs==3:
                a1l_3t.append(1)
                jet_in_1l_3t.append(JET.count(1))
            else: 
		a1l_3t.append(0)
		jet_in_1l_3t.append(0)  
	    ####
            if numbMuons+numbEls<=1 and numb_with_tauhs==0:
                a01l_0t.append(1)
                jet_in_01l_0t.append(JET.count(1))
            else: 
		a01l_0t.append(0)
		jet_in_01l_0t.append(0)
            ####
	    if numbMuons+numbEls==2 and numb_with_tauhs==0:
                a2l_0t.append(1)
                jet_in_2l_0t.append(JET.count(1))
            else: 
		a2l_0t.append(0)
		jet_in_2l_0t.append(0)
            ####
	    if numbMuons+numbEls==3 and numb_with_tauhs==0:
                a3l_0t.append(1)
                jet_in_3l_0t.append(JET.count(1))
            else: 
		a3l_0t.append(0)
		jet_in_3l_0t.append(0)
            ####
	    if numbMuons+numbEls==2 and numb_with_tauhs==1:
                a2l_1t.append(1)
                jet_in_2l_1t.append(JET.count(1))
            else: 
		a2l_1t.append(0)
		jet_in_2l_1t.append(0)
            ####
	    if numbMuons+numbEls==1 and numb_with_tauhs==2:
                a1l_2t.append(1)
                jet_in_1l_2t.append(JET.count(1))
            else: 
		a1l_2t.append(0)
		jet_in_1l_2t.append(0)
            ####
	    if numbMuons+numbEls==5 and numb_with_tauhs==1:
                a5l_1t.append(1)
                jet_in_5l_1t.append(JET.count(1))
            else: 
		a5l_1t.append(0)
		jet_in_5l_1t.append(0)
            ####
	    if numbMuons+numbEls==4 and numb_with_tauhs==2:
                a4l_2t.append(1)
                jet_in_4l_2t.append(JET.count(1))
            else: 
		a4l_2t.append(0)
		jet_in_4l_2t.append(0)
            ####
	    if numbMuons+numbEls==0 and numb_with_tauhs==2:
                a0l_2t.append(1)
                jet_in_0l_2t.append(JET.count(1))
            else: 
		a0l_2t.append(0)
		jet_in_0l_2t.append(0)

            ##################### Category 
              
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
            
            #for i in range(0,branchParticle.GetEntries()):
            br_Hs[0]=int(len(GenHs))
            br_Es[0]=int(len(GenEs))
            br_Ms[0]=len(GenMs)
            HH.Fill()            
            E.Fill()
            M.Fill()
 
            br_countEvent[0]=a8l_0t[entry]
            br_Njets[0]=jet_in_8l_0t[entry]
            a.Fill()

            br_countEvent[0]=a3l_1t[entry]
            br_Njets[0]=jet_in_3l_1t[entry]
            b.Fill()

            br_countEvent[0]=a2l_2t[entry]
            br_Njets[0]=jet_in_2l_2t[entry]
            c.Fill()

            br_countEvent[0]=a1l_3t[entry]
            br_Njets[0]=jet_in_1l_3t[entry]
            d.Fill()

            br_countEvent[0]=a01l_0t[entry]
            br_Njets[0]=jet_in_01l_0t[entry]
            e.Fill()
     	    
            br_countEvent[0]=a2l_0t[entry]
            br_Njets[0]=jet_in_2l_0t[entry]
            m.Fill()

            br_countEvent[0]=a3l_0t[entry]
            br_Njets[0]=jet_in_3l_0t[entry]
            p.Fill()

            br_countEvent[0]=a2l_1t[entry]
            br_Njets[0]=jet_in_2l_1t[entry]
            h.Fill()

	    br_countEvent[0]=a1l_2t[entry]
	    br_Njets[0]=jet_in_1l_2t[entry]
	    ii.Fill()

	    br_countEvent[0]=a5l_1t[entry]
	    br_Njets[0]=jet_in_5l_1t[entry]
	    jj.Fill()

	    br_countEvent[0]=a4l_2t[entry]
	    br_Njets[0]=jet_in_4l_2t[entry]
	    k.Fill()

	    br_countEvent[0]=a0l_2t[entry]
	    br_Njets[0]=jet_in_0l_2t[entry]
	    l.Fill()
            
            
            histNumb_of_jets_8l_0t.Fill(jet_in_8l_0t[entry])
            histNumb_of_jets_3l_1t.Fill(jet_in_3l_1t[entry])
	    histNumb_of_jets_2l_2t.Fill(jet_in_2l_2t[entry])
	    histNumb_of_jets_1l_3t.Fill(jet_in_1l_3t[entry])
	    histNumb_of_jets_01l_0t.Fill(jet_in_01l_0t[entry])
	    histNumb_of_jets_2l_0t.Fill(jet_in_2l_0t[entry])
	    histNumb_of_jets_3l_0t.Fill(jet_in_3l_0t[entry])
	    histNumb_of_jets_2l_1t.Fill(jet_in_2l_1t[entry])
	    histNumb_of_jets_1l_2t.Fill(jet_in_1l_2t[entry])
	    histNumb_of_jets_5l_1t.Fill(jet_in_5l_1t[entry])
	    histNumb_of_jets_4l_2t.Fill(jet_in_4l_2t[entry])
            histNumb_of_jets_0l_2t.Fill(jet_in_0l_2t[entry])
                
#########################
#fi.Close()
print "processed "+str(totEvt)+" "

tot_numb_8l_0t=np.sum(a8l_0t) ###### total number of events for category l>=4_0tauh
tot_numb_3l_1t=np.sum(a3l_1t) ###### total number of events for category 3l_1tauh
tot_numb_2l_2t=np.sum(a2l_2t) ###### total number of events for category 2l_2tauh
tot_numb_1l_3t=np.sum(a1l_3t) ###### total number of events for category 1l_3tauh
tot_numb_01l_0t=np.sum(a01l_0t) ###### total number of events for category l<=1_0tauh
tot_numb_2l_0t=np.sum(a2l_0t) ###### total number of events for category 2l_0tauh
tot_numb_3l_0t=np.sum(a3l_0t) ###### total number of events for category 3l_0tauh
tot_numb_2l_1t=np.sum(a2l_1t) ###### total number of events for category 2l_1tauh
tot_numb_1l_2t=np.sum(a1l_2t) ###### total number of events for category 1l_2tauh
tot_numb_5l_1t=np.sum(a5l_1t) ###### total number of events for category 5l_1tauh
tot_numb_4l_2t=np.sum(a4l_2t) ###### total number of events for category 4l_2tauh
tot_numb_0l_2t=np.sum(a0l_2t) ###### total number of events for category 0l_2tauh


#print("total number of events for category l>=4_0tauh",tot_numb_8l_0t)
#print("total number of events for category 3l_1tauh",tot_numb_3l_1t)
#print("total number of events for category 2l_2tauh",tot_numb_2l_2t)
#print("total number of events for category 1l_3tauh",tot_numb_1l_3t)
#print("total number of events for category l<=1_0tauh",tot_numb_01l_0t)
#print("total number of events for category 2l_0tauh",tot_numb_2l_0t)
#print("total number of events for category 3l_0tauh",tot_numb_3l_0t)
#print("total number of events for category 2l_1tauh",tot_numb_2l_1t)
#print("total number of events for category 1l_2tauh",tot_numb_1l_2t)
#print("total number of events for category 5l_1tauh",tot_numb_5l_1t)
#print("total number of events for category 4l_2tauh",tot_numb_4l_2t)
#print("total number of events for category 0l_2tauh",tot_numb_0l_2t)


#print ("TOTAL number of jets in category l>=4_0tauh",np.sum(jet_in_8l_0t))
#print ("TOTAL number of jets in category 3l_1tauh",np.sum(jet_in_3l_1t))
#print ("TOTAL number of jets in category 2l_2tauh",np.sum(jet_in_2l_2t))
#print ("TOTAL number of jets in category 1l_3tauh",np.sum(jet_in_1l_3t))
#print ("TOTAL number of jets in category l<=1_0tauh",np.sum(jet_in_01l_0t))
#print ("TOTAL number of jets in category 2l_0tauh",np.sum(jet_in_2l_0t))
#print ("TOTAL number of jets in category 3l_0tauh",np.sum(jet_in_3l_0t))
#print ("TOTAL number of jets in category 2l_1tauh",np.sum(jet_in_2l_1t))
#print ("TOTAL number of jets in category 1l_2tauh",np.sum(jet_in_1l_2t))
#print ("TOTAL number of jets in category 5l_1tauh",np.sum(jet_in_5l_1t))
#print ("TOTAL number of jets in category 4l_2tauh",np.sum(jet_in_4l_2t))
#print ("TOTAL number of jets in category 0l_2tauh",np.sum(jet_in_0l_2t))


numb_jet_in_8l_0t=np.sum(jet_in_8l_0t)
numb_jet_in_3l_1t=np.sum(jet_in_3l_1t)
numb_jet_in_2l_2t=np.sum(jet_in_2l_2t)
numb_jet_in_1l_3t=np.sum(jet_in_1l_3t)
numb_jet_in_01l_0t=np.sum(jet_in_01l_0t)
numb_jet_in_2l_0t=np.sum(jet_in_2l_0t)
numb_jet_in_3l_0t=np.sum(jet_in_3l_0t)
numb_jet_in_2l_1t=np.sum(jet_in_2l_1t)
numb_jet_in_1l_2t=np.sum(jet_in_1l_2t)
numb_jet_in_5l_1t=np.sum(jet_in_5l_1t)
numb_jet_in_4l_2t=np.sum(jet_in_4l_2t)
numb_jet_in_0l_2t=np.sum(jet_in_0l_2t)


#for i  in range(0,tot_numb_8l_0t):
#    histNumb_of_jets_8l_0t.Fill(numb_jet_in_8l_0t)

#for i  in range(0,tot_numb_3l_1t):
#    histNumb_of_jets_3l_1t.Fill(numb_jet_in_3l_1t)

#for i  in range(0,tot_numb_2l_2t):
#    histNumb_of_jets_2l_2t.Fill(numb_jet_in_2l_2t)

#for i  in range(0,tot_numb_1l_3t):
#    histNumb_of_jets_1l_3t.Fill(numb_jet_in_1l_3t)

#for i  in range(0,tot_numb_01l_0t):
#    histNumb_of_jets_01l_0t.Fill(numb_jet_in_01l_0t)

#for i  in range(0,tot_numb_2l_0t):
#    histNumb_of_jets_2l_0t.Fill(numb_jet_in_2l_0t)

#for i  in range(0,tot_numb_3l_0t):
#    histNumb_of_jets_3l_0t.Fill(numb_jet_in_3l_0t)

#for i  in range(0,tot_numb_2l_1t):
#    histNumb_of_jets_2l_1t.Fill(numb_jet_in_2l_1t)

#for i  in range(0,tot_numb_1l_2t):
#    histNumb_of_jets_1l_2t.Fill(numb_jet_in_1l_2t)

#for i  in range(0,tot_numb_5l_1t):
#    histNumb_of_jets_5l_1t.Fill(numb_jet_in_5l_1t)

#for i  in range(0,tot_numb_4l_2t):
#    histNumb_of_jets_4l_2t.Fill(numb_jet_in_4l_2t)

#for i  in range(0,tot_numb_0l_2t):
#    histNumb_of_jets_0l_2t.Fill(numb_jet_in_0l_2t)
################################## filling the branch

#for i in range(0,len(a8l_0t)):
#    br_countEvent[0]=int(tot_numb_8l_0t)
#    br_Njets[0]=jet_in_8l_0t[i]
#    a.Fill()
#for i in range(0,len(a3l_1t)):
#    br_countEvent[0]=int(tot_numb_3l_1t)
#    br_Njets[0]=jet_in_3l_1t[i]
#    b.Fill()
#for i in range(0,len(a2l_2t)):
#    br_countEvent[0]=int(tot_numb_2l_2t)
#    br_Njets[0]=jet_in_2l_2t[i]
#    c.Fill()
#for i in range(0,len(a1l_3t)):
#    br_countEvent[0]=int(tot_numb_1l_3t)
#    br_Njets[0]=jet_in_1l_3t[i]
#    d.Fill()
#for i in range(0,len(a01l_0t)):
#    br_countEvent[0]=int(tot_numb_01l_0t)
#    br_Njets[0]=jet_in_01l_0t[i]
#    e.Fill()
#for i in range(0,len(a2l_0t)):
#    br_countEvent[0]=int(tot_numb_2l_0t)
#    br_Njets[0]=jet_in_2l_0t[i]
#    m.Fill()
#for i in range(0,len(a3l_0t)):
#    br_countEvent[0]=int(tot_numb_3l_0t)
#    br_Njets[0]=jet_in_3l_0t[i]
#    p.Fill()
#for i in range(0,len(a2l_1t)):
#    br_countEvent[0]=int(tot_numb_2l_1t)
#    br_Njets[0]=jet_in_2l_1t[i]
#    h.Fill()
#for i in range(0,len(a1l_2t)):
#    br_countEvent[0]=int(tot_numb_1l_2t)
#    br_Njets[0]=jet_in_1l_2t[i]
#    ii.Fill()
#for i in range(0,len(a5l_1t)):
#    br_countEvent[0]=int(tot_numb_5l_1t)
#    br_Njets[0]=jet_in_5l_1t[i]
#    jj.Fill()
#for i in range(0,len(a4l_2t)):
#    br_countEvent[0]=int(tot_numb_4l_2t)
#    br_Njets[0]=jet_in_4l_2t[i]
#    k.Fill()
#for i in range(0,len(a0l_2t)):
#    br_countEvent[0]=int(tot_numb_0l_2t)
#    br_Njets[0]=jet_in_0l_2t[i]
#    l.Fill()


##################### drawing
c2=histMass_W.Draw()
c3=histMass_Z.Draw()
histMass_T.Draw()

#fc=ROOT.TFile.Open('Categories.root','read')
canv1=TCanvas('RESULT1',"",400,400)
#canv2=TCanvas("RESULT2","",400,400)
#canv3=TCanvas("RESULT3","",400,400)
#canv4=TCanvas("RESULT4","",400,400)
#canv5=TCanvas("RESULT5","",400,400)
#canv6=TCanvas("RESULT6","",400,400)
#canv7=TCanvas("RESULT7","",400,400)
#canv8=TCanvas("RESULT8","",400,400)
#canv9=TCanvas("RESULT9","",400,400)
#canv10=TCanvas("RESULT10","",400,400)
#canv11=TCanvas("RESULT11","",400,400)
#canv12=TCanvas("RESULT12","",400,400)

canv1.cd()
histNumb_of_jets_8l_0t.Draw()
canv1.Print("results.pdf(","pdf")
histNumb_of_jets_3l_1t.Draw()
canv1.Print("results.pdf(","pdf")
histNumb_of_jets_2l_2t.Draw()
canv1.Print("results.pdf","pdf")
histNumb_of_jets_1l_3t.Draw()
canv1.Print("results.pdf","pdf")
histNumb_of_jets_01l_0t.Draw()
canv1.Print("results.pdf","pdf")
histNumb_of_jets_2l_0t.Draw()
canv1.Print("results.pdf","pdf")
histNumb_of_jets_3l_0t.Draw()
canv1.Print("results.pdf","pdf")
histNumb_of_jets_2l_1t.Draw()
canv1.Print("results.pdf","pdf")
histNumb_of_jets_1l_2t.Draw()
canv1.Print("results.pdf","pdf")
histNumb_of_jets_5l_1t.Draw()
canv1.Print("results.pdf","pdf")
histNumb_of_jets_4l_2t.Draw()
canv1.Print("results.pdf","pdf")
histNumb_of_jets_0l_2t.Draw()
canv1.Print("results.pdf)","pdf")

#f.close()
#raw_input()


if not onlyCount :
    for i in range(0,len(C)):
        out_file = ROOT.TFile(os.getcwd()+'/Folder_HHTo4T/'+'Out_'+str(C[i]), 'RECREATE')    ####    out_file = ROOT.TFile("teste111.root", 'RECREATE')
        out_file.WriteTObject(tuple, tuple.GetName(), 'Overwrite')
        #out_file.WriteTObject(HH,HH.GetName(),'Overwrite')
        #out_file.WriteTObject(Ms,Ms.GetName(),'Overwrite')
        #out_file.WriteTObject(Es,Es.GetName(),'overwrite')
        out_file.Write()
        out_file.Close()
    oout_file=ROOT.TFile('Categories.root','RECREATE')
    oout_file.WriteTObject(a, a.GetName(), 'Overwrite')
    oout_file.WriteTObject(b, b.GetName(), 'Overwrite')
    oout_file.WriteTObject(c, c.GetName(), 'Overwrite')
    oout_file.WriteTObject(d, d.GetName(), 'Overwrite')
    oout_file.WriteTObject(e, e.GetName(), 'Overwrite')
    oout_file.WriteTObject(m, m.GetName(), 'Overwrite')
    oout_file.WriteTObject(p, p.GetName(), 'Overwrite')
    oout_file.WriteTObject(h, h.GetName(), 'Overwrite')
    oout_file.WriteTObject(ii, ii.GetName(), 'Overwrite')
    oout_file.WriteTObject(jj, jj.GetName(), 'Overwrite')
    oout_file.WriteTObject(k, k.GetName(), 'Overwrite')
    oout_file.WriteTObject(l, l.GetName(), 'Overwrite')
    oout_file.Write()
    oout_file.Close()


#sys.stdout=orig_stdout
#f.close()
#############################################################################################################################               
#print("Number of events with 5 Taus  ",N_Of_Ev_5tau.count(1))
#print ("Number of tau pairs", len(ifTaus))
#print ("Number of  W pairs", len(ifWs))
#print ("Number of Z pairs",len(ifZs))
###################### Hist. of masses of 2 Taus /// 2 Bozons
#print ("Number of needed Tau pairs", ifTaus.count(1))
#print ("Number of needed W pairs",ifWs.count(1))
#print("Number of needed Z pairs",ifZs.count(1))
#print("length of HH_TT / HH_WW // HH_ZZ ",len(HH_TT),len(HH_WW),len(HH_ZZ)) 

########################   YIELDS
#BR_HHTo4T=0.0039337984
#BR_HHTo4V=0.0463559
#BR_HHTo2T2V=0.01504586
#sigma_SM=33.86
#Lum=3000
#sigma_TOT=(sigma_SM)*(BR_HHTo4T)
#if "HHTo4T" in inputFile:
#    sigma_TOT=(sigma_SM)*(BR_HHTo4T)
    
#if "HHTo4V" in inputFile:
#    sigma_TOT=(sigma_SM)*(BR_HHTo4V)
    
#if "HHTo2T2V" in inputFile:
#    sigma_TOT=(sigma_SM)*(BR_HHTo2T2V)
   
#Yield_8l_0t=((tot_numb_8l_0t)*(sigma_TOT)*(Lum))/1000
#Yield_3l_1t=((tot_numb_3l_1t)*(sigma_TOT)*(Lum))/1000
#Yield_2l_2t=((tot_numb_2l_2t)*(sigma_TOT)*(Lum))/1000
#Yield_1l_3t=((tot_numb_1l_3t)*(sigma_TOT)*(Lum))/1000
#Yield_01l_0t=((tot_numb_01l_0t)*(sigma_TOT)*(Lum))/1000
#Yield_2l_0t=((tot_numb_2l_0t)*(sigma_TOT)*(Lum))/1000
#Yield_3l_0t=((tot_numb_3l_0t)*(sigma_TOT)*(Lum))/1000
#Yield_2l_1t=((tot_numb_2l_1t)*(sigma_TOT)*(Lum))/1000
#Yield_1l_2t=((tot_numb_1l_2t)*(sigma_TOT)*(Lum))/1000
#Yield_5l_1t=((tot_numb_5l_1t)*(sigma_TOT)*(Lum))/1000
#Yield_4l_2t=((tot_numb_4l_2t)*(sigma_TOT)*(Lum))/1000
#Yield_0l_2t=((tot_numb_0l_2t)*(sigma_TOT)*(Lum))/1000

#print "Yield_8l_0t = ",Yield_8l_0t 
#print "Yield_3l_1t = ",Yield_3l_1t
#print "Yield_2l_2t = ",Yield_2l_2t
#print "Yield_1l_3t = ",Yield_1l_3t
#print "Yield_01l_0t = ",Yield_01l_0t
#print "Yield_2l_0t = ",Yield_2l_0t
#print "Yield_3l_0t = ",Yield_3l_0t
#print "Yield_2l_1t = ",Yield_2l_1t
#print "Yield_1l_2t = ",Yield_1l_2t
#print "Yield_5l_1t = ",Yield_5l_1t
#print "Yield_4l_2t = ",Yield_4l_2t
#print "Yield_0l_2t = ",Yield_0l_2t

#################################### FOLDER

#sys.stdout=orig_stdout
raw_input()


