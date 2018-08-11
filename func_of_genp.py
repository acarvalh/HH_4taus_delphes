def gen_matching(b,v,t,h,e,m,tt,ww,zz,vectZ,vectT,vectW,dumb):
    if IsPU==0 and pdgCode==13 and genparticle.Status==1:
        #print ("(pdgCode, genparticle.Status)____",pdgCode, genparticle.Status)
        m.append(genparticle)
    if IsPU==0 and pdgCode==11 and genparticle.Status==1:
        #print ("(pdgCode, genparticle.Status)____",pdgCode, genparticle.Status)
        e.append(genparticle)
    if IsPU == 0 and pdgCode == 25 and genparticle.Status==22:
        #print ("(pdgCode, genparticle.Status)____",pdgCode, genparticle.Status)
        h.append(genparticle)
    if IsPU == 0 and abs(pdgCode) == 24  and genparticle.Status == 22 :
        #print (pdgCode, genparticle.Status)
        v.append(genparticle)
        vectW=ROOT.TLorentzVector()
        vectW.SetPtEtaPhiM(genparticle.PT,genparticle.Eta,genparticle.Phi,genparticle.Mass)
        ww.append(vectW)
    if IsPU == 0 and abs(pdgCode) == 23  and genparticle.Status == 22 :
        #print (pdgCode, genparticle.Status)
        v.append(genparticle)
        vectZ=ROOT.TLorentzVector()
        vectZ.SetPtEtaPhiM(genparticle.PT,genparticle.Eta,genparticle.Phi,genparticle.Mass)
        zz.append(vectZ)
    if IsPU == 0 and (abs(pdgCode) == 15) and genparticle.Status == 2 :
        #print ("(pdgCode, genparticle.Status)____",pdgCode, genparticle.Status)
        t.append(genparticle)
        vectT=ROOT.TLorentzVector()
        vectT.SetPtEtaPhiM(genparticle.PT,genparticle.Eta,genparticle.Phi,genparticle.Mass)
        tt.append(vectT)
        #print charge
        #print ("in this event there are ",len(HH_TT) ,"particles")
    if IsPU == 0 and abs(pdgCode) == 5  and genparticle.Status == 23 :
        #print (pdgCode, genparticle.Status)
        if genparticle.PT > 10 :
            dumb = ROOT.TLorentzVector()
            dumb.SetPtEtaPhiM(genparticle.PT,genparticle.Eta,genparticle.Phi,genparticle.Mass)
            b.append(dumb)

    #return ("number of Higgses / taus / V's / b's / E's / M's", len(GenHs), len(GenTaus), len(GenVs), len(GenBs), len(GenEs),len(GenMs))

def fill_particles_info(
    category,
    taus, taus_charge,
    muons, muons_charge,
    els, electrons_charge,
    recoJets, recoJetsBtag
    ) :
    for pp in xrange(8) :
        ## ["pt", "eta", "phi", "mass", "charge"]
        if pp < len(tauhs) :
            dict_trees[category][2][0][pp] = taus[pp].Pt()
            dict_trees[category][2][1][pp] = taus[pp].Eta()
            dict_trees[category][2][2][pp] = taus[pp].Phi()
            dict_trees[category][2][3][pp] = taus[pp].M()
            dict_trees[category][2][4][pp] = taus_charge[pp]
        else :
            dict_trees[category][2][0][pp] = -100.
            dict_trees[category][2][1][pp] = -100.
            dict_trees[category][2][2][pp] = -100.
            dict_trees[category][2][3][pp] = -100.
            dict_trees[category][2][4][pp] = -100.
        #########################################
        if pp < len(Muons) :
            dict_trees[category][3][0][pp] = muons[pp].Pt()
            dict_trees[category][3][1][pp] = muons[pp].Eta()
            dict_trees[category][3][2][pp] = muons[pp].Phi()
            dict_trees[category][3][3][pp] = muons[pp].M()
            dict_trees[category][3][4][pp] = muons_charge[pp]
        else :
            dict_trees[category][3][0][pp] = -100.
            dict_trees[category][3][1][pp] = -100.
            dict_trees[category][3][2][pp] = -100.
            dict_trees[category][3][3][pp] = -100.
            dict_trees[category][3][4][pp] = -100.
        #########################################
        if pp < len(els) :
            dict_trees[category][4][0][pp] = els[pp].Pt()
            dict_trees[category][4][1][pp] = els[pp].Eta()
            dict_trees[category][4][2][pp] = els[pp].Phi()
            dict_trees[category][4][3][pp] = els[pp].M()
            dict_trees[category][4][4][pp] = electrons_charge[pp]
        else :
            dict_trees[category][4][0][pp] = -100.
            dict_trees[category][4][1][pp] = -100.
            dict_trees[category][4][2][pp] = -100.
            dict_trees[category][4][3][pp] = -100.
            dict_trees[category][4][4][pp] = -100.
        #########################################
        if pp < len(recoJets) :
            dict_trees[category][5][0][pp] = recoJets[pp].Pt()
            dict_trees[category][5][1][pp] = recoJets[pp].Eta()
            dict_trees[category][5][2][pp] = recoJets[pp].Phi()
            dict_trees[category][5][3][pp] = recoJets[pp].M()
            dict_trees[category][5][4][pp] = recoJetsBtag[pp]
        else :
            dict_trees[category][5][0][pp] = -100.
            dict_trees[category][5][1][pp] = -100.
            dict_trees[category][5][2][pp] = -100.
            dict_trees[category][5][3][pp] = -100.
            dict_trees[category][5][4][pp] = -100.
