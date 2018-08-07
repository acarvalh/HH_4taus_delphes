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
nn=[]
ax=[]
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
#fi.Write()
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

histNumb_of_jets=ROOT.TH1D("number of jets in category 2l-tauh","number of jets in category 2l-tauh",50,0,10)
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
        execfile("bjos_trying.py")
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
            countMUpair=0
            countElpair=0
            countJetpair=0
            count_Muon=0
            count_Elect=0
            Muons=[]
            Els=[]
            jetss=[]
          
            #print c
            ############################################################  M U O N
            for part in range(0,branchMuonLoose.GetEntries()):
                muon=branchMuonLoose.At(part)
                if muon.PT>9 and abs(muon.Eta)<2.4: 
                    Muons.append(1)
            numbMuons=np.sum(Muons)
            
            ############################################################# E L E C T R O N
            for part in range(0,branchElectron.GetEntries()):
                electron=branchElectron.At(part)
                if electron.PT>13 and abs(electron.Eta)<2.5: 
                    Els.append(1)
                else: Els.append(0)
	    numbEls=np.sum(Els)
            
            #print np.sum(Els)
            ############################################################## J E T 

            for part in range(0, branchJet.GetEntries()): # add one more collection to the delphes card
                jet =  branchJet.At(part)
                if jet.PT > 20 and abs(jet.Eta)<2.3 :
                    jetss.append(1)
                else: jetss.append(0)
	    numbjetss=np.sum(jetss)
            #####################    Category 2l_2tauh 
            #ax=[]
            if numbMuons==2 and numbjetss==2 or numbEls==2 and numbjetss==2 or numbEls==1 and numbMuons==1 and numbjetss==2:
                nn.append(1)
                ax.append(len(jetss))
            else: 
		nn.append(0)
		ax.append(0)
            
            #print ("number of jets in this category", np.sum(ax))
            
            #br_countEvent[0]=nn
            #c.Fill()
            
            #print ("electron pair number/# of events with 2 electrons========",countElpair)

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
#fi.Close()
print "processed "+str(totEvt)+" "

nnn=np.sum(nn) ###### total number of events for category 2l_2tauh
print ("TOTAL number of jets in category 2l-2tauh",np.sum(ax))
print len(ax)
for i  in range(0,len(ax)):
    histNumb_of_jets.Fill(ax[i])
    
histNumb_of_jets.Draw()


for i in range(0,len(nn)):
    br_countEvent[0]=int(nnn)
    c.Fill()
fi.Write()
#fi.Close()
c2=histMass_W.Draw()
c3=histMass_Z.Draw()
histMass_T.Draw()
histNumb_of_jets.Draw()
raw_input()
if not onlyCount :
    for i in range(0,len(C)):
        out_file = ROOT.TFile(os.getcwd()+'/Folder_HHTo4T/'+'Out_'+str(C[i]), 'RECREATE')    ####    out_file = ROOT.TFile("teste111.root", 'RECREATE')
        out_file.WriteTObject(tuple, tuple.GetName(), 'Overwrite')
        out_file.WriteTObject(HH,HH.GetName(),'Overwrite')
        out_file.WriteTObject(Ms,Ms.GetName(),'Overwrite')
        out_file.WriteTObject(Es,Es.GetName(),'overwrite')
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
#print ("electron pair number/# of events with 2 electrons========",countElpair)
#print ("Number of needed Tau pairs", ifTaus.count(1))
#print ("Number of needed W pairs",ifWs.count(1))
#print("Number of needed Z pairs",ifZs.count(1))
#print("length of HH_TT / HH_WW // HH_ZZ ",len(HH_TT),len(HH_WW),len(HH_ZZ)) 

########################


#################################### FOLDER

#sys.stdout=orig_stdout
raw_input()


