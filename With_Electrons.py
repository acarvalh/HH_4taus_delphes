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
from collections import OrderedDict
execfile("functions.py")
# Delphes headers
ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gInterpreter.Declare('#include "DelphesClasses.h"')
ROOT.gSystem.Load("libDelphes")
from collections import OrderedDict
from optparse import OptionParser
parser = OptionParser()
parser.add_option("--input ", type="string", dest="input", help="A valid file in Delphes format. If you give somthing without .root extension it will interpret as a folder and try to process all the root files on that folder in series", default=" /eos/user/a/acarvalh/delphes_HH_4tau/HHTo4VTo4L/tree_HHTo4VTo4L_10.root")
parser.add_option("--onlyCount", action="store_true", dest="onlyCount", help="Only reports the number of events on the sample", default=False)
parser.add_option("--mass", type="float", dest="mass", help="eventual sample-by sample cut", default=125.)
parser.add_option("--nameOut", type="int", dest="nameOut", help="tag output file", default=0)
parser.add_option("--totalevt", type="int", dest="totalevt", help="total number of events on all avaiable samples", default=0)
(options, args) = parser.parse_args()

inputFile = options.input
toProcess = [str(inputFile)]
print str(inputFile)
if "*" in str(inputFile) :
    print str(inputFile)
    toProcess = glob.glob(str(inputFile))
print ("the first sample is: ", toProcess[0], " of a total of ",len(toProcess),"samples")
#print(inputFile)
file = open(os.getcwd()+'/samplesList.txt',"w")
file.write(str(glob.glob(str(inputFile)+'/*.root')))
file.close()
#######################  Output file
s=[]
#orig_stdout=sys.stdout
#A=inputFile.split("/")
#B=str(A[-1])
#C=tree_3l_1tauh.split(".")
#print C[0]
#f=open(os.getcwd()+'/Folder_HHTo4T/'+'str(B[0])'+'_Out'+'.root',"w")
#f = open(os.getcwd()+'/Folder_HHTo4T/'+str(C[0])+'_Out'+'.txt', 'w')
#sys.stdout = f
########################### TREE OF CATEGORIES
## ge = greater or equal
## le = less or equal
Categories=[
    'tree_lge4_0tauh',
    'tree_3l_1tauh',
    'tree_2l_2tauh',
    'tree_1l_3tauh',
    'tree_1l_0tauh',
    'tree_0l_0tauh',
    'tree_0l_4tauh',          #### just added
    'tree_0l_3tauh',          #### just added
    'tree_0l_5getauh',        #### just added
    'tree_2los_0tauh',
    'tree_2lss_0tauh',
    'tree_3l_0tauh',
    'tree_2l_1tauh',
    'tree_1l_2tauh',
    'tree_5l_1tauh',
    'tree_4l_2tauh',
    'tree_0l_2tauh',
    'tree_1l_1tauh',       ### missed ones
    'tree_0l_1tauh',       ### missed ones
    'tree_3l_2tauh',       ### missed ones
    'tree_1l_4tauh',       ### missed ones
    'tree_2l_3tauh',       ### missed ones
    'tree_4l_3tauh',       ### missed ones
    'tree_4l_1tauh',       ### missed ones
    'tree_3l_3tauh',       ### missed ones
    'tree_2l_4tauh',       ### missed ones
    'tree_1l_5tauh',       ### missed ones
    #'tree_total'
    ]
print("number of categories___",len(Categories))
dict_trees = OrderedDict()
branches_int_names = ["evtcounter", "njets", "nBs" , "nelectrons",  "nmuons", "ntaus", "sum_lep_charge", "sum_tauh_charge"]
branches_double_arrays_names = ["pt", "eta", "phi", "mass", "charge"]
branches_double_array_name=["invmass"]
branches_double_names = ["MissingET", "MissingET_eta", "MissingET_phi", "scalarHT"]

for cat in Categories :
    dict_trees[cat] = [
        ROOT.TTree(cat, cat),
        [ array( 'i', [0] ) for i in range(len(branches_int_names))], ## save integers
        [ array( 'd', 8*[ 0. ] ) for i in range(len(branches_double_arrays_names))], ## save arrays of up to 8 taus
        [ array( 'd', 8*[ 0. ] ) for i in range(len(branches_double_arrays_names))], ## save arrays of up to 8 muons
        [ array( 'd', 8*[ 0. ] ) for i in range(len(branches_double_arrays_names))], ## save arrays of up to 8 electrons
        [ array( 'd', 8*[ 0. ] ) for i in range(len(branches_double_arrays_names))], ## save arrays of up to 8 jets
        [ array( 'd', [0] ) for i in range(len(branches_double_names))], ## save integers
        [ array( 'd', [0. ] ) for i in range(len(branches_double_array_name))]  ####################### zemotas mdzime movashoro amas tu wavshi !!!
        ]
    ## adding the common branches to all the categories
    for bb, branch in enumerate(branches_int_names) : dict_trees[cat][0].Branch(branch, dict_trees[cat][1][bb], branch+'/I')
    for ii, info in enumerate(["pt", "eta", "phi", "mass", "charge"]) :
        dict_trees[cat][0].Branch("tau_"+info, dict_trees[cat][2][ii], "tau_"+info+'[8]/D')
        dict_trees[cat][0].Branch("muon_"+info, dict_trees[cat][3][ii], "muon_"+info+'[8]/D')
        dict_trees[cat][0].Branch("electron_"+info, dict_trees[cat][4][ii], "electron_"+info+'[8]/D')
        dict_trees[cat][0].Branch("jet_"+info, dict_trees[cat][5][ii], "jet_"+info+'[8]/D')
    for bb, branch in enumerate(branches_double_names) : dict_trees[cat][0].Branch(branch, dict_trees[cat][6][bb], branch+'/D')
    if "2tauh" in cat:
        dict_trees[cat][0].Branch("tauhs_invmass", dict_trees[cat][7][0],"tauhs_invmass"+'/D')
    if "2l" in cat:
	dict_trees[cat][0].Branch("lepton_invmass", dict_trees[cat][7][0],"lepton_invmass"+'/D')


## Chalenge: adding and filling a branch specific to a category
## add the invariant mass of a pair of ebjects weneaver there are 2 objects for the bellow categories:
# 'tree_2l_0tauh' / 'tree_2l_1tauh' / 'tree_1l_2tauh' / 'tree_4l_2tauh'
# tip: with dict_trees['tree_2l_0tauh'] we access the information of 'tree_2l_0tauh'

########################   matching
execfile("Matching_Taus.py")
execfile("Matching_Ws.py")
execfile("Matching_Zs.py")
execfile("func_of_genp.py") # gen_matching, fill_particles_info
#execfile("Treeee.py")
#########################
# Cuts
#########################
execfile("cuts.py")
#########################
# Doing the trees with events
#########################
# declare branches
#tree_noCuts=ROOT.TTree('tree_noCuts',"tree_noCuts")
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
Electron=tuple.Branch('Es',br_Es,'Es/I')
Muon=tuple.Branch('Ms',br_Ms,'Ms/I')
#############################################################
# Loop over file list
#############################################################
onlyCount = options.onlyCount
#C=[] ### try to use an inteligible name (preferably with more than one letter)
## We only want to keep track of the name of the file if we will be running the analysis code file by file
## (you will understand better that when we will be processing the BKG)
## See the fix on this looking on the FIXME_FILENAME

totEvt = 0
nJets = 0
nBs = 0
Hs=0
Es=0
Ms=0
Weights = 1
Tau21 = -1.
N_Of_Ev_5tau=[]

countEvt = 0
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
        #branchJet = treeReader.UseBranch("JetPUPPI")
        branchParticle = treeReader.UseBranch("Particle")
        #branchPhotonLoose = treeReader.UseBranch("PhotonLoose")
        #branchPhotonTight = treeReader.UseBranch("PhotonTight")
        branchMuonLoose = treeReader.UseBranch("MuonLoose")  #
        #branchMuonTight = treeReader.UseBranch("MuonTight")
        branchElectron = treeReader.UseBranch("Electron")    #
        #branchMuonLooseCHS = treeReader.UseBranch("MuonLooseCHS")
        #branchMuonTightCHS = treeReader.UseBranch("MuonTightCHS")
        #branchElectronCHS = treeReader.UseBranch("ElectronCHS")
        branchMissingET = treeReader.UseBranch("MissingET")
        branchPuppiMissingET = treeReader.UseBranch("PuppiMissingET")
        branchScalarHT = treeReader.UseBranch("ScalarHT")
        ##execfile("bjos_trying.py")
    	##B=str(A[-1])
    	##C=tree_3l_1tauh.split(".")
    	##print C[0]
        #branchEvent = treeReader.UseBranch("Event")
        #branchJet = treeReader.UseBranch("Jet")
        #branchParticle = treeReader.UseBranch("Particle")
        for entry in range(0, numberOfEntries):
            #print "entry = "+str(entry)+" ================================================================="
            # Load selected branches with data from specified event
            countEvt+=1
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
                gen_matching(GenBs,GenVs,GenTaus,GenHs,GenEs,GenMs,HH_TT,HH_WW,HH_ZZ,vectZ,vectT,vectW,dumb)
                #print ("number of Higgses / taus / V's / b's / E's / M's", len(GenHs), len(GenTaus), len(GenVs), len(GenBs), len(GenEs),len(GenMs))
            #########################
            if len(HH_TT)==5 and len(GenHs)==2:
                N_Of_Ev_5tau.append(1)
            else:
                N_Of_Ev_5tau.append(0)
            if len(GenHs) != 2 :
                print "not two H's"
                break
            #### why 5 ???
            #if len(GenVs)!=4 :
            #    print "Not 4 Bosons"
            #########################
            muon_charge=[]
            electron_charge=[]
            tauhs_charge=[]
            #tauhs_mass=[]
            Muons=[]  ###### if yes, 1...if no 0
            Els=[]    ###### if yes, 1...if no 0
            tauhs=[]  ###### if yes, 1...if no 0

            #### Implement
            # findTauPairFromH
            # findTauPairFromV
            RecoJets = []
            RecoJetsBtag = []
            #RecoBJets = []
            for part in range(0, branchJet.GetEntries()): # add one more collection to the delphes card
                jet =  branchJet.At(part)
                ### first test if it is a tau tag
                if jet.TauTag and jet.PT > 20 and abs(jet.Eta)<2.3 :
                    vecttau=ROOT.TLorentzVector()
                    vecttau.SetPtEtaPhiM(jet.PT,jet.Eta,jet.Phi,jet.Mass)
                    tauhs.append(vecttau)
                    tauhs_charge.append(jet.Charge)
                    #tauhs_mass.append(jet.Mass)
                elif jet.PT > 25 and abs(jet.Eta) < 2.5 : ## check on jet eta cut
                    dumb = ROOT.TLorentzVector()
                    dumb.SetPtEtaPhiM(jet.PT,jet.Eta,jet.Phi,jet.Mass)
                    RecoJets.append(dumb)
                    RecoJetsBtag.append(jet.BTag)
                    #RecoBJets.append(isbtagged(dumb, GenBs))
                   ## using the DR with the genParticles to find out if there is a b-quark
            #tauhs_inv_mass=np.sum(tauhs_mass)
            numbb = 0
            for i in range(0, len(RecoJetsBtag)) : numbb += RecoJetsBtag[i];
            nJets = len(RecoJets)
            nBs = numbb
            #print("it must be anumber of B-s in each event____",nBs)
            numb_with_tauhs=len(tauhs)    ######## number of those tauhs in each event
            if numb_with_tauhs==2:
		tauhsmm=(tauhs[0]+tauhs[1]).M()
                #print tauhs_mass
                #print tauhs_inv_mass
            #######################
            br_nJets[0] = int(nJets)
            br_nBs[0] = int(nBs)
            tuple.Fill()
            ######################## CATEGORIES
            ############################################################  M U O N
            for part in range(0,branchMuonLoose.GetEntries()):
                muon=branchMuonLoose.At(part)
                if muon.PT>9 and abs(muon.Eta)<2.4:
                    vectmuon=ROOT.TLorentzVector(muon.PT,muon.Eta,muon.Phi,muon.T)
                    Muons.append(vectmuon)
                    muon_charge.append(muon.Charge)
            numbMuons=len(Muons)  ######## number of those muons in each event
            ############################################################# E L E C T R O N
            for part in range(0,branchElectron.GetEntries()):
                electron=branchElectron.At(part)
                if electron.PT>13 and abs(electron.Eta)<2.5:
                    vectelect=ROOT.TLorentzVector(electron.PT,electron.Eta,electron.Phi,electron.T)
                    Els.append(vectelect)
                    electron_charge.append(electron.Charge)
            numbEls=len(Els)   ######## number of those electrons in each event
            ##################### counting events in Categories
            branches_int_fill = [
                1, #"evtcounter",
                nJets, #"njets" ,
                nBs, # "nBs",
                numbEls, #"nelectrons",
                numbMuons, #"nmuons",
                numb_with_tauhs, # ntaus
                sum(muon_charge) + sum(electron_charge),  #"sum_lep_charge",
                sum(tauhs_charge) #"sum_tauh_charge"
                ]
            branches_double_fill = [
                branchMissingET.At(0).MET, #"MissingET",
                branchMissingET.At(0).Eta, # "MissingET_eta",
                branchMissingET.At(0).Phi, # "MissingET_phi",
                branchScalarHT.At(0).HT # "scalarHT"
                ]
            #branches_double_array_name_fill =[
            #    tauhs_inv_mass
            #    ]
            passedOneCategory = 0 ### to ensure that the categories do not overlap
            if numbMuons+numbEls>=4 and numb_with_tauhs==0 :
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_lge4_0tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_lge4_0tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_lge4_0tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_lge4_0tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==3 and numb_with_tauhs==1 :
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_3l_1tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_3l_1tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_3l_1tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_3l_1tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==2 and numb_with_tauhs==2:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_2l_2tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_2l_2tauh'][6][bb][0] = branches_double_fill[bb]
                #for mm,branch in enumerate(branch_double_aaray_name) : dict_trees['tree_2l_2tauh'][7][mm][0]=branches_double_array_name_fill[mm]
                dict_trees['tree_2l_2tauh'][7][0]=tauhsmm
		#print tauhsmm
                #print (tauhs[0]+tauhs[1]).M()
                fill_particles_info('tree_2l_2tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_2l_2tauh'][0].Fill()
		####dict_trees['tree_0l_2tauh'][7][0]=lep_inv_mass
                passedOneCategory+=1
            if (numbMuons + numbEls)==1 and numb_with_tauhs==3:
                for bb, branch in enumerate(branches_int_names) :  dict_trees['tree_1l_3tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_1l_3tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_1l_3tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_1l_3tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==1 and numb_with_tauhs==0: ### review that category deffinition
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_1l_0tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_1l_0tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_1l_0tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_1l_0tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==0 and numb_with_tauhs==0: ### review that category deffinition
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_0l_0tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_0l_0tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_0l_0tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_0l_0tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==0 and numb_with_tauhs==4:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_0l_4tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_0l_4tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_0l_4tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_0l_4tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==0 and numb_with_tauhs==3:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_0l_3tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_0l_3tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_0l_3tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_0l_3tauh'][0].Fill()
		passedOneCategory+=1
            if (numbMuons + numbEls)==0 and numb_with_tauhs>=5:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_0l_5getauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_0l_5getauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_0l_5getauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_0l_5getauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==2 and numb_with_tauhs==0 and np.sum(electron_charge)==0:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_2los_0tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_2los_0tauh'][6][bb][0] = branches_double_fill[bb]
		####dict_trees['tree_0l_2tauh'][7][0]=lep_inv_mass
                fill_particles_info('tree_2los_0tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_2los_0tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==2 and numb_with_tauhs==0 and np.sum(electron_charge)!=0:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_2lss_0tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_2lss_0tauh'][6][bb][0] = branches_double_fill[bb]
		####dict_trees['tree_0l_2tauh'][7][0]=lep_inv_mass
                fill_particles_info('tree_2lss_0tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_2lss_0tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==3 and numb_with_tauhs==0:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_3l_0tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_3l_0tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_3l_0tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_3l_0tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==2 and numb_with_tauhs==1:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_2l_1tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_2l_1tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_2l_1tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                ####dict_trees['tree_0l_2tauh'][7][0]=lep_inv_mass
                dict_trees['tree_2l_1tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==1 and numb_with_tauhs==2:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_1l_2tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_1l_2tauh'][6][bb][0] = branches_double_fill[bb]
                #for mm,branch in enumerate(branch_double_aaray_name) : dict_trees['tree_1l_2tauh'][7][mm][0]=branches_double_array_name_fill[mm]
		dict_trees['tree_1l_2tauh'][7][0]=tauhsmm
                fill_particles_info('tree_1l_2tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_1l_2tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==5 and numb_with_tauhs==1:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_5l_1tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_5l_1tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_5l_1tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_5l_1tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons+numbEls)==4 and numb_with_tauhs==2:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_4l_2tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_4l_2tauh'][6][bb][0] = branches_double_fill[bb]
                #for mm,branch in enumerate(branch_double_aaray_name) : dict_trees['tree_4l_2tauh'][7][mm][0]=branches_double_array_name_fill[mm]
                dict_trees['tree_4l_2tauh'][7][0]=tauhsmm
                dict_trees['tree_4l_2tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons+numbEls)==0 and numb_with_tauhs==2:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_0l_2tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_0l_2tauh'][6][bb][0] = branches_double_fill[bb]
                #for mm,branch in enumerate(branch_double_aaray_name) : dict_trees['tree_0l_2tauh'][7][mm][0]=branches_double_array_name_fill[mm]
                dict_trees['tree_0l_2tauh'][7][0]=tauhsmm
                fill_particles_info('tree_0l_2tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_0l_2tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==0 and numb_with_tauhs==1:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_0l_1tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_0l_1tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_0l_1tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_0l_1tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==1 and numb_with_tauhs==1:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_1l_1tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_1l_1tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_1l_1tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_1l_1tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==3 and numb_with_tauhs==2:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_3l_2tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_3l_2tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_3l_2tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_3l_2tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==1 and numb_with_tauhs==4:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_1l_1tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_1l_4tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_1l_4tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_1l_4tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==2 and numb_with_tauhs==3:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_2l_3tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_2l_3tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_2l_3tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_2l_3tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==4 and numb_with_tauhs==3:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_4l_3tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_4l_3tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_4l_3tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_4l_3tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==4 and numb_with_tauhs==1:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_4l_1tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_4l_1tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_4l_1tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_4l_1tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==3 and numb_with_tauhs==3:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_3l_3tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_3l_3tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_3l_3tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_3l_3tauh'][0].Fill()
		passedOneCategory+=1
            if (numbMuons + numbEls)==2 and numb_with_tauhs==4:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_2l_4tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_2l_4tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_2l_4tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_2l_4tauh'][0].Fill()
                passedOneCategory+=1
            if (numbMuons + numbEls)==1 and numb_with_tauhs==5:
                for bb, branch in enumerate(branches_int_names) : dict_trees['tree_1l_5tauh'][1][bb][0] = branches_int_fill[bb]
                for bb, branch in enumerate(branches_double_names) : dict_trees['tree_1l_5tauh'][6][bb][0] = branches_double_fill[bb]
                fill_particles_info('tree_1l_5tauh',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
                dict_trees['tree_1l_5tauh'][0].Fill()
                passedOneCategory+=1
            if passedOneCategory > 1 :
                print "The categorries overlap!!!"
                break
            #elif passedOneCategory == 1 : ## if passed one category count as total
            #    for bb, branch in enumerate(branches_int_names) : dict_trees['tree_total'][1][bb][0] = branches_int_fill[bb]
            #    for bb, branch in enumerate(branches_double_names) : dict_trees['tree_total'][6][bb][0] = branches_double_fill[bb]
            #    fill_particles_info('tree_total',  tauhs, tauhs_charge,  Muons, muon_charge, Els, electron_charge, RecoJets, RecoJetsBtag)
            #    dict_trees['tree_total'][0].Fill()

	    #if passedOneCategory==0 :
	    #	s.append(1)
            #    print ("number of leptons", (numbMuons+numbEls),"           ", "number of tauhs", numb_with_tauhs)
            #else: s.append(0)


            ########################   Matching 2 Taus / 2V-s
            ifTaus=[]
            ifWs=[]
            ifZs=[]
            for i in range(0,branchParticle.GetEntries()):
                HisMatchedTaus(HH_TT,ifTaus)
                HisMatchedWs(HH_WW,ifWs)
                HisMatchedZs(HH_ZZ,ifZs)
            ######################## Filling Branches
            #i in range(0,branchParticle.GetEntries()):
            br_Hs[0]=int(len(GenHs))
            br_Es[0]=int(len(GenEs))
            br_Ms[0]=len(GenMs)
            HH.Fill()
            Electron.Fill()
            Muon.Fill()
#########################
print "processed "+str(totEvt)+" "
#########################             MINE
#output_labels=[]
#for i in range(0,len(toProcess)):
#    filein=toProcess[i].split('/')
#    output_labels.append(filein[-1])             ####### for example: [...,...,..,tree_HHTo4T_220.root,....,...,..]
#########################
channel = "default"
if "HHTo4T" in inputFile : channel = "HHTo4T"
if "HHTo4V" in inputFile : channel = "HHTo4V"
if "HHTo2T2V" in inputFile : channel = "HHTo2T2V"
base_folder=os.getcwd()+'/Folder_'+channel+'/'
if not onlyCount :
    ### FIXME_FILENAME
    #########################              MINE

    #    base_folder = os.getcwd()+'/Folder_'+channel+'/'         ########## for example: ..../Folder_HHTo4T/
    #    if not os.path.exists(base_folder) : os.mkdir( base_folder )
    #    for i in range(0,len(output_labels)):
    #        nameout = base_folder+'Out_'+str(output_labels[i])   ########## for example: ..../Folder_HHTo4T/tree_HHTo4T_220.root
    #        out_file = ROOT.TFile(nameout, 'RECREATE')
    #        out_file.WriteTObject(tuple, tuple.GetName())
    ##########################
    #output_label = 1
    #if len(toProcess) == 1 :                    ####when we are running separetly
    #    filein=toProcess[0].split("/")
    #    output_label = filein[-1]       #### f.g. tree_HHTo4T_220.root
    #base_folder = os.getcwd()+'/Folder_'+channel+'/'
    #if not os.path.exists(base_folder) : os.mkdir( base_folder )
    #nameout = base_folder+'Out_'+str(output_label)
    #out_file = ROOT.TFile(nameout, 'RECREATE')
    #out_file.WriteTObject(tuple, tuple.GetName())
    ### you can save the tree categories on the same file
    #for cat in Categories : out_file.WriteTObject(dict_trees[cat][0], dict_trees[cat][0].GetName(), 'Overwrite') ###  but after 4 spaces also in mine
    #out_file.Write()          ### also in mine
    #out_file.Close()          ### also in mine
    #print ("saved", nameout)       ### also in mine
    if options.nameOut == 0 : name="result"
    else : name="part_"+str(options.nameOut)
    nameOut=base_folder+'Out_'+name+'.root'
    out_file=ROOT.TFile(nameOut,'RECREATE')
    out_file.WriteTObject(tuple,tuple.GetName())
    for cat in Categories: out_file.WriteTObject(dict_trees[cat][0], dict_trees[cat][0].GetName(), 'Overwrite')
    out_file.Write()
    out_file.Close()
    print("saved",nameOut)
    ########################   YIELDS
    BR_HHTo4T=0.0039337984
    BR_HHTo4V=0.0463559
    BR_HHTo2T2V=0.01504586
    sigma_SM=33.86
    Lum=3000
    sigma_TOT=(sigma_SM)*(BR_HHTo4T)
    if "HHTo4T" in inputFile : sigma_TOT=(sigma_SM)*(BR_HHTo4T)
    if "HHTo4V" in inputFile : sigma_TOT=(sigma_SM)*(BR_HHTo4V)
    if "HHTo2T2V" in inputFile : sigma_TOT=(sigma_SM)*(BR_HHTo2T2V)
    print "countEvt____Out_____"+str(countEvt)
    denominator = 1
    if options.nameOut == 0 : ### do table of yields only if I know I had ran on all the events -- to BKG the solution will be other
        denominator = countEvt
        namefiletext = base_folder+"yields_"+channel+".txt"
        fileWrite = open(namefiletext, "w")
        fileWrite.write("category totalEvt Nevt Eff Yield \n")
        for cat in Categories :
            eff_cat = float(dict_trees[cat][0].GetEntries())/float(denominator)
            yield_cat = eff_cat*sigma_TOT*Lum
            print("total (number of events / yields) for category ", cat, int(dict_trees[cat][0].GetEntries()), eff_cat, yield_cat)
            fileWrite.write(
            cat+" "+str(denominator)+" "+str(int(dict_trees[cat][0].GetEntries()))+\
            " "+str(eff_cat)+" "+str(yield_cat)+"\n"
            )
            ### the 'tree_lle1_0tauh' is very populated, we may like to separate it as 'tree_1l_0tauh' and 'tree_0l_0tauh'
        ### challenge: add some lines counting the number of oposite sign and same sign leptons/tauh
        # weneaver there are 2 objects for the bellow:
        # 'tree_2l_0tauh' / 'tree_2l_1tauh' / 'tree_1l_2tauh' / 'tree_4l_2tauh'
        ##
        # Tip: bellow 'tree_0l_2tauh' is separated by charge
        #print dict_trees['tree_0l_2tauh'][0].GetEntries( "sum_tauh_charge == 0")
        #print dict_trees['tree_0l_2tauh'][0].GetEntries( "sum_tauh_charge != 0")
        fileWrite.close()
        print ("saved", namefiletext)
else :
    namefiletext = base_folder+"AllCount_"+channel+".txt"
    fileWrite = open(namefiletext, "w")
    fileWrite.write("processed "+str(totEvt)+" ")
    fileWrite.close()
    print ("saved", namefiletext)
#print ("the number of missed ones________", np.sum(s))
