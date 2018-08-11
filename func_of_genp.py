def g(b,v,t,h,e,m,tt,ww,zz,vectZ,vectT,vectW,dumb):
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
