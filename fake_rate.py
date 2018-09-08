if WP==32:                #### I used the numbers listed in the final table ..before the numbers were (2.05, 1.80, 1.00)
   fake_factor_3=1.96
   fake_factor_2o3=1.8
   fake_factor_1o4=1.64


for part in range(0,branchJet.GetEntries()):
    jet=branchJet.At(part)
    pasTauID=False
    passJet=False
    if jet.Tau>20 and Weight > 0.1 and jet.PT > 20 and abs(jet.Eta)<2.8:
        ......
        if passTauID:
            ......
	    tauhs.append(vecttau)
    if not passTauID and jet.PT > 25 and abs(jet.Eta) < 2.5:   ###I think I should change abs(jet.Eta)<2.5 at least to the abs(jet.Eta)<3.0 , no ?
        fake_rate_abs=-8.33753e-03+1.48065e-03*jet.PT-3.23176e-05*pow(jet.PT,2)+2.91151e-07*pow(jet.PT,3)-1.20285e-09*(jet.PT,4)
	fake_rate_barr=0
        if jet.PT>190: fake_rate_abs+=1.88459e-12*(jet.PT,5)*0.25
        if jet.PT<190: fake_rate_abs+=0.00058
	if jet.Eta>1.95: fake_rate_barr=1.84916*(jet.Eta-1.4)    ### relative to barrel
	if jet.Eta<1.95: fake_rate_barr=1                        ### relative to barrel
        test_rate= random.uniform(0.0,1.0)
        if jet.Eta<3.0 and jet.Eta>2.3 and test_rate  < (fake_rate_abs)*(fake_factor_3) : passJet = True     #### or should I use jet.Eta < 2.8 after all ?
        if jet.Eta<=2.3 and jet.Eta>1.4 and test_rate <(fake_rate_barr)*(fake_factor_2o3) : passJet=True
        if jet.Eta<=1.4 and test_rate<(fake_rate_barr)*(ake_rate_factor_1o4): passJet=True
        if passJet:
            .....
	    RecoJets.append(dumb)

