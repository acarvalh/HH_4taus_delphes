#!/bin/bash


finalState="4V"
cmsbase="/afs/cern.ch/work/a/acarvalh/delphes_main/Delphes_CMS/CMSSW_8_1_0/src/"
localFolder="/afs/cern.ch/work/a/acarvalh/delphes_main/HH_4taus_delphes/"
channelFoldersubmit="Folder_HHTo"$finalState"_submit"
channelFolderResult="Folder_HHTo"$finalState"/"

# Out_part_9.root

mkdir $channelFoldersubmit
### here you have the total count of events -- as double check
exec 3<> $channelFoldersubmit/submit_HH_signal_totalCount.txt
    echo "#!/bin/bash" >&3
    echo "cd "$cmsbase >&3
    echo "eval \`scram runtime -sh\`" >&3
    echo "cd "$localFolder >&3
    echo "python With_Electrons.py --input \"/eos/cms/store/user/acarvalh/delphes_HH_4tau/HHTo"$finalState"/*\" --onlyCount " >&3
# Close file
exec 3>&-
#cd $channelFoldersubmit
#bsub  -q 8nm -J jobCount < submit_HH_signal_totalCount.txt
#cd $localFolder

## here you process ~10 files by time on batch
for ii in {10..40}
do
  fileOut=$channelFolderResult"Out_part_"$ii".root"
  if [[ -f $fileOut && -s $fileOut ]]; then
    echo $fileOut" exist and not empty"
  else
    echo "doing submit "$ii
    exec 3<> $channelFoldersubmit/submit_HH_signal_$ii.txt
        echo "#!/bin/bash" >&3
        echo "cd "$cmsbase >&3
        echo "eval \`scram runtime -sh\`" >&3
        echo "cd "$localFolder >&3
        echo "python With_Electrons.py --input \"/eos/cms/store/user/acarvalh/delphes_HH_4tau/HHTo"$finalState"/*_"$ii"*.root\"  --nameOut "$ii >&3
    exec 3>&-
    cd $channelFoldersubmit
    bsub  -q 8nm -J job$ii < submit_HH_signal_$ii.txt
    cd $localFolder
  fi
done

## here you process ~1 files by time on batch
for ii in {5..9}
do
  fileOut=$channelFolderResult"Out_part_"$ii".root"
  if [[ -f $fileOut && -s $fileOut ]]; then
    echo $fileOut" exist and not empty"
  else
    echo "doing submit "$ii
    exec 3<> $channelFoldersubmit/submit_HH_signal_$ii.txt
        echo "#!/bin/bash" >&3
        echo "cd "$cmsbase >&3
        echo "eval \`scram runtime -sh\`" >&3
        echo "cd "$localFolder >&3
        echo "python With_Electrons.py --input \"/eos/cms/store/user/acarvalh/delphes_HH_4tau/HHTo"$finalState"/*_"$ii"*.root\"  --nameOut "$ii >&3
    exec 3>&-
    cd $channelFoldersubmit
    bsub  -q 8nm -J job$ii < submit_HH_signal_$ii.txt
    cd $localFolder
  fi
done

## here you process ~1 files by time on batch
for ii in {1..4}
do
  fileOut=$channelFolderResult"Out_part_"$ii".root"
  if [[ -f $fileOut && -s $fileOut ]]; then
    echo $fileOut" exist and not empty"
  else
    echo "doing submit "$ii
    exec 3<> $channelFoldersubmit/submit_HH_signal_$ii.txt
        echo "#!/bin/bash" >&3
        echo "cd "$cmsbase >&3
        echo "eval \`scram runtime -sh\`" >&3
        echo "cd "$localFolder >&3
        echo "python With_Electrons.py --input \"/eos/cms/store/user/acarvalh/delphes_HH_4tau/HHTo"$finalState"/*_"$ii".root\"  --nameOut "$ii >&3
    exec 3>&-
    cd $channelFoldersubmit
    bsub  -q 8nm -J job$ii < submit_HH_signal_$ii.txt
    cd $localFolder
  fi
done
