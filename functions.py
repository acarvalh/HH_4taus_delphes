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

def read_channel(Categories, channelLocal, totalEvtLocal) :
    sumAllTest = 0
    toRead = glob.glob(str(options.input)+"/Folder_"+channelLocal+"/*.root")
    dict_data = OrderedDict()
    for category in Categories :
        dict_data[category] = [pandas.DataFrame(columns=["evtcounter"])]
        for file in toRead :
            try: tfile = ROOT.TFile(file)
            except :
            	print ('file ', file,' is corrupt')
            	continue
            try: tree = tfile.Get(category)
            except :
            	print ('Tree for category ', category,' is corrupt')
            	continue
            if tree is not None :
                try : chunk_arr = tree2array(tree, branches=featuresToPlot)
                except : continue # tree was empty
                else :
                    chunk_df = pandas.DataFrame(chunk_arr) #
                    ### we can manipulate some branches here
                    #chunk_df["MissingET_eta"]=abs(chunk_df["MissingET_eta"])
                    #chunk_df["MissingET_phi"]=abs(chunk_df["MissingET_phi"])
                    dict_data[category][0]=dict_data[category][0].append(chunk_df, ignore_index=True)
            else : print ("file "+file+"was empty")
            tfile.Close()
        print ("Category ",category, "of channel", channelLocal,  "had ", len(dict_data[category][0]["evtcounter"]),"events")
        sumAllTest+=len(dict_data[category][0]["evtcounter"])
    print ("read",sumAllTest, "events, and we should have ",totalEvtLocal," events ",100*(float(sumAllTest)/float(totalEvtLocal)),"% were processed")
    return dict_data
