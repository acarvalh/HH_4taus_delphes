#!/bin/bash


background="ttbar"
cmsbase="/afs/cern.ch/user/t/tsupatas/CMSSW_8_1_0/src/"
localFolder="/afs/cern.ch/user/t/tsupatas/CMSSW_8_1_0/src/HH_4taus_delphes/"
channelFoldersubmit="Folder_BKG"$background"_submit"
channelFolderResult="Folder_BKG"$background"/"

mkdir $channelFoldersubmit
mkdir $channelFolderResult


for ii in {1..9}
do
  fileOut=$channelFolderResult"Out_part_"$ii".root"
  if [[ -f $fileOut && -s $fileOut ]]; then
    echo $fileOut" exist and not empty"
  else
    echo "doing submit "$ii
    exec 3<> $channelFoldersubmit/submit_HH_BKGttbar_$ii.txt
        echo "#!/bin/bash" >&3
        echo "cd "$cmsbase >&3
        echo "eval \`scram runtime -sh\`" >&3
        echo "cd "$localFolder >&3
        echo "python With_Electrons.py --input \"/eos/cms/store/group/upgrade/delphes_output/YR_Delphes/Delphes342pre15/TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU/TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_"$ii"_*.root\"  --nameOut "$ii >&3
    exec 3>&-
    cd $channelFoldersubmit
    bsub  -q 8nm -J job$ii < submit_HH_BKGttbar_$ii.txt
    cd $localFolder
  fi
done


for ii in {10..39}
do
  fileOut=$channelFolderResult"Out_part_"$ii".root"
  if [[ -f $fileOut && -s $fileOut ]]; then
    echo $fileOut" exist and not empty"
  else
    echo "doing submit "$ii
    exec 3<> $channelFoldersubmit/submit_HH_BKGttbar_$ii.txt
        echo "#!/bin/bash" >&3
        echo "cd "$cmsbase >&3
        echo "eval \`scram runtime -sh\`" >&3
        echo "cd "$localFolder >&3
        echo "python With_Electrons.py --input \"/eos/cms/store/group/upgrade/delphes_output/YR_Delphes/Delphes342pre15/TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU/TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_"$ii"*.root\"  --nameOut "$ii >&3
    exec 3>&-
    cd $channelFoldersubmit
    bsub  -q 8nh -J job$ii < submit_HH_BKGttbar_$ii.txt
    cd $localFolder
  fi
done


for ii in {40..99}
do
  fileOut=$channelFolderResult"Out_part_"$ii".root"
  if [[ -f $fileOut && -s $fileOut ]]; then
    echo $fileOut" exist and not empty"
  else
    echo "doing submit "$ii
    exec 3<> $channelFoldersubmit/submit_HH_BKGttbar_$ii.txt
        echo "#!/bin/bash" >&3
        echo "cd "$cmsbase >&3
        echo "eval \`scram runtime -sh\`" >&3
        echo "cd "$localFolder >&3
        echo "python With_Electrons.py --input \"/eos/cms/store/group/upgrade/delphes_output/YR_Delphes/Delphes342pre15/TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_200PU/TT_TuneCUETP8M2T4_14TeV-powheg-pythia8_"$ii"*.root\"  --nameOut "$ii >&3
    exec 3>&-
    cd $channelFoldersubmit
    bsub  -q 8nh -J job$ii < submit_HH_BKGttbar_$ii.txt
    cd $localFolder
  fi
done


