def isbtagged(jets, GenB) :
    #print "calculate DR"
    see=0
    for bjets in GenB :
        if not bjets.Pt() > 0 or not jets.Pt() > 0 :
            if bjets.DeltaR(jets) < 0.3 : see = see + 1
        #else : print "The problem is here "+str(bjets.Pt())+" "+str(jets.Pt())
    if see > 0 : return see
    else : return 0

sign = lambda a: 1 if a>0 else -1 if a<0 else 0


