function OPENCL_FULL {
    ./opencl_full > $1/opencl_fullEP.txt profileEntertainmentPolitical.bin
    ./opencl_full > $1/opencl_fullEF.txt profileEntertainmentFinancial.bin
    ./opencl_full > $1/opencl_fullEI.txt profileEntertainmentInternational.bin
    ./opencl_full > $1/opencl_fullEW.txt profileEntertainmentWashington.bin
    ./opencl_full > $1/opencl_fullSP.txt profileSportsPolitical.bin
    ./opencl_full > $1/opencl_fullSF.txt profileSportsFinancial.bin
    ./opencl_full > $1/opencl_fullSI.txt profileSportsInternational.bin
    ./opencl_full > $1/opencl_fullSW.txt profileSportsWashington.bin
    ./opencl_full > $1/opencl_fullUP.txt profileUSAPolitical.bin
    ./opencl_full > $1/opencl_fullUF.txt profileUSAFinancial.bin
    ./opencl_full > $1/opencl_fullUI.txt profileUSAInternational.bin
    ./opencl_full > $1/opencl_fullUW.txt profileUSAWashington.bin
}

function OPENCL_FULL_FORK {
    ./opencl_full_fork > $1/opencl_full_forkEP.txt profileEntertainmentPolitical.bin
    ./opencl_full_fork > $1/opencl_full_forkEF.txt profileEntertainmentFinancial.bin
    ./opencl_full_fork > $1/opencl_full_forkEI.txt profileEntertainmentInternational.bin
    ./opencl_full_fork > $1/opencl_full_forkEW.txt profileEntertainmentWashington.bin
    ./opencl_full_fork > $1/opencl_full_forkSP.txt profileSportsPolitical.bin
    ./opencl_full_fork > $1/opencl_full_forkSF.txt profileSportsFinancial.bin
    ./opencl_full_fork > $1/opencl_full_forkSI.txt profileSportsInternational.bin
    ./opencl_full_fork > $1/opencl_full_forkSW.txt profileSportsWashington.bin
    ./opencl_full_fork > $1/opencl_full_forkUP.txt profileUSAPolitical.bin
    ./opencl_full_fork > $1/opencl_full_forkUF.txt profileUSAFinancial.bin
    ./opencl_full_fork > $1/opencl_full_forkUI.txt profileUSAInternational.bin
    ./opencl_full_fork > $1/opencl_full_forkUW.txt profileUSAWashington.bin
}
# Fara has the CPU and GPU we're testing with
if [ "$HOSTNAME" = "fara.dcs.gla.ac.uk" ]; then
    # No Bloom GPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO 
    mkdir -p htmlResultsWaterlooNoBloom/GPU
    echo "Testing full system without bloom filter on GPU"
    OPENCL_FULL "htmlResultsWaterlooNoBloom/GPU"
    # No Bloom CPUGPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO 
    mkdir -p htmlResultsWaterlooNoBloom/CPUGPU
    echo "Testing full system without bloom filter on CPU and GPU"
    OPENCL_FULL_FORK "htmlResultsWaterlooNoBloom/CPUGPU"
    # No Bloom CPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DDEVCPU
    mkdir -p htmlResultsWaterlooNoBloom/CPU
    echo "Testing full system without bloom filter on CPU"
    OPENCL_FULL "htmlResultsWaterlooNoBloom/CPU"
    # Bloom GPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilter.raw\"'
    mkdir -p htmlResultsWaterlooBloom/GPU
    echo "Testing full system with bloom filter on GPU"
    OPENCL_FULL "htmlResultsWaterlooBloom/GPU"
    # Bloom CPUGPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilter.raw\"'
    mkdir -p htmlResultsWaterlooBloom/CPUGPU
    echo "Testing full system with bloom filter on CPU and GPU"
    OPENCL_FULL_FORK "htmlResultsWaterlooBloom/CPUGPU"
    # Bloom CPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilter.raw\"' CPPFLAGS+=-DDEVCPU
    mkdir -p htmlResultsWaterlooBloom/CPU
    echo "Testing full system with bloom filter on CPU"
    OPENCL_FULL "htmlResultsWaterlooBloom/CPU"
    # BloomAll0 GPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll0.raw\"'
    mkdir -p htmlResultsWaterlooBloomAll0/GPU
    echo "Testing full system with bloom filter all zeroes on GPU"
    OPENCL_FULL "htmlResultsWaterlooBloomAll0/GPU"
    # BloomAll0 CPUGPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll0.raw\"'
    mkdir -p htmlResultsWaterlooBloomAll0/CPUGPU
    echo "Testing full system with bloom filter all zeroes on CPU and GPU"
    OPENCL_FULL_FORK "htmlResultsWaterlooBloomAll0/CPUGPU"
    # BloomAll0 CPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll0.raw\"' CPPFLAGS+=-DDEVCPU
    mkdir -p htmlResultsWaterlooBloomAll0/CPU
    echo "Testing full system with bloom filter all zeroes on CPU"
    OPENCL_FULL "htmlResultsWaterlooBloomAll0/CPU"
    # BloomAll1 GPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll1.raw\"'
    mkdir -p htmlResultsWaterlooBloomAll1/GPU
    echo "Testing full system with bloom filter all ones on GPU"
    OPENCL_FULL "htmlResultsWaterlooBloomAll1/GPU"
    # BloomAll1 CPUGPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll1.raw\"'
    mkdir -p htmlResultsWaterlooBloomAll1/CPUGPU
    echo "Testing full system with bloom filter all ones on CPU and GPU"
    OPENCL_FULL_FORK "htmlResultsWaterlooBloomAll1/CPUGPU"
    # BloomAll1 CPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll1.raw\"' CPPFLAGS+=-DDEVCPU
    mkdir -p htmlResultsWaterlooBloomAll1/CPU
    echo "Testing full system with bloom filter all ones on CPU"
    OPENCL_FULL "htmlResultsWaterlooBloomAll1/CPU"
fi
# Curieuse GTX 590
if [ "$HOSTNAME" = "curieuse" ]; then
    # No Bloom 1x GTX 590
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO 
    mkdir -p htmlResultsWaterlooNoBloom/HalfGTX590
    echo "Testing full system without bloom filter on one half of 590"
    OPENCL_FULL "htmlResultsWaterlooNoBloom/HalfGTX590"
    # Bloom 1x GTX 590
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilter.raw\"'
    mkdir -p htmlResultsWaterlooBloom/HalfGTX590
    echo "Testing full system with bloom filter on one half of 590"
    OPENCL_FULL "htmlResultsWaterlooBloom/HalfGTX590"
    # BloomAll0 1x GTX 590
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll0.raw\"'
    mkdir -p htmlResultsWaterlooBloomAll0/HalfGTX590
    echo "Testing full system with all 0 bloom filter on one half of 590"
    OPENCL_FULL "htmlResultsWaterlooBloomAll0/HalfGTX590"
    # Bloom 1x GTX 590
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll1.raw\"'
    mkdir -p htmlResultsWaterlooBloomAll1/HalfGTX590
    echo "Testing full system with all 1 bloom filter on one half of 590"
    OPENCL_FULL "htmlResultsWaterlooBloomAll1/HalfGTX590"
    # No Bloom Full GTX 590
    make clean
    make CPPFLAGS+=-DDEVACC CPPFLAGS+=-DPHIPHI CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DGPUGPU
    mkdir -p htmlResultsWaterlooNoBloom/FullGTX590
    echo "Testing full system without bloom filter on full 590"
    OPENCL_FULL_FORK "htmlResultsWaterlooNoBloom/FullGTX590"
    # Bloom Full GTX 590
    make clean
    make CPPFLAGS+=-DDEVACC CPPFLAGS+=-DPHIPHI CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilter.raw\"' CPPFLAGS+=-DGPUGPU
    mkdir -p htmlResultsWaterlooBloom/FullGTX590
    echo "Testing full system with bloom filter on full 590"
    OPENCL_FULL_FORK "htmlResultsWaterlooBloom/FullGTX590"
    # BloomAll0 Full GTX 590
    make clean
    make CPPFLAGS+=-DDEVACC CPPFLAGS+=-DPHIPHI CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll0.raw\"' CPPFLAGS+=-DGPUGPU
    mkdir -p htmlResultsWaterlooBloomAll0/FullGTX590
    echo "Testing full system with all 0 bloom filter on full 590"
    OPENCL_FULL_FORK "htmlResultsWaterlooBloomAll0/FullGTX590"
    # Bloom Full GTX 590
    make clean
    make CPPFLAGS+=-DDEVACC CPPFLAGS+=-DPHIPHI CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll1.raw\"' CPPFLAGS+=-DGPUGPU
    mkdir -p htmlResultsWaterlooBloomAll1/FullGTX590
    echo "Testing full system with all 1 bloom filter on full 590"
    OPENCL_FULL_FORK "htmlResultsWaterlooBloomAll1/FullGTX590"
fi
# Manipa has the Intel Phi we're testing with
if [ "$HOSTNAME" = "manipa" ]; then
    # No Bloom Phi
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DDEVACC
    mkdir -p htmlResultsWaterlooNoBloom/Phi
    echo "Testing full system without bloom filter on Phi"
    OPENCL_FULL "htmlResultsWaterlooNoBloom/Phi"
    # No Bloom PhiPhi
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DDEVACC CPPFLAGS+=-DPHIPHI
    mkdir -p htmlResultsWaterlooNoBloom/PhiPhi
    echo "Testing full system without bloom filter on Phi and Phi"
    OPENCL_FULL_FORK "htmlResultsWaterlooNoBloom/PhiPhi"
    # Bloom Phi
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DDEVACC CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilter.raw\"'
    mkdir -p htmlResultsWaterlooBloom/Phi
    echo "Testing full system with bloom filter on Phi"
    OPENCL_FULL "htmlResultsWaterlooBloom/Phi"
    # Bloom PhiPhi
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DDEVACC CPPFLAGS+=-DPHIPHI CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilter.raw\"'
    mkdir -p htmlResultsWaterlooBloom/PhiPhi
    echo "Testing full system with bloom filter on Phi and Phi"
    OPENCL_FULL_FORK "htmlResultsWaterlooBloom/PhiPhi"
    # BloomAll0 Phi
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DDEVACC CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll0.raw\"'
    mkdir -p htmlResultsWaterlooBloomAll0/Phi
    echo "Testing full system with bloom filter all zeroes on Phi"
    OPENCL_FULL "htmlResultsWaterlooBloomAll0/Phi"
    # BloomAll0 PhiPhi
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DDEVACC CPPFLAGS+=-DPHIPHI CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll0.raw\"'
    mkdir -p htmlResultsWaterlooBloomAll0/PhiPhi
    echo "Testing full system with bloom filter all zeroes on Phi and Phi"
    OPENCL_FULL_FORK "htmlResultsWaterlooBloomAll0/PhiPhi"
    # BloomAll1 Phi
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DDEVACC CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll1.raw\"'
    mkdir -p htmlResultsWaterlooBloomAll1/Phi
    echo "Testing full system with bloom filter all ones on Phi"
    OPENCL_FULL "htmlResultsWaterlooBloomAll1/Phi"
    # BloomAll1 PhiPhi
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DDEVACC CPPFLAGS+=-DPHIPHI CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll1.raw\"'
    mkdir -p htmlResultsWaterlooBloomAll1/PhiPhi
    echo "Testing full system with bloom filter all ones on Phi and Phi"
    OPENCL_FULL_FORK "htmlResultsWaterlooBloomAll1/PhiPhi"
fi
# Togian has a 64 core CPU
if [ "$HOSTNAME" = "togian.dcs.gla.ac.uk" ]; then
    # No Bloom CPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DDEVCPU
    mkdir -p htmlResultsWaterlooNoBloom/AMDCPU
    echo "Testing full system without bloom filter on CPU"
    OPENCL_FULL "htmlResultsWaterlooNoBloom/AMDCPU"
    # Bloom CPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilter.raw\"' CPPFLAGS+=-DDEVCPU
    mkdir -p htmlResultsWaterlooBloom/AMDCPU
    echo "Testing full system with bloom filter on CPU"
    OPENCL_FULL "htmlResultsWaterlooBloom/AMDCPU"
    # BloomAll0 CPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll0.raw\"' CPPFLAGS+=-DDEVCPU
    mkdir -p htmlResultsWaterlooBloomAll0/AMDCPU
    echo "Testing full system with bloom filter all zeroes on CPU"
    OPENCL_FULL "htmlResultsWaterlooBloomAll0/AMDCPU"
    # BloomAll1 CPU
    make clean
    make CPPFLAGS+=-DHTML_PARSE  CPPFLAGS+=-DWATERLOO  CPPFLAGS+=-DBLOOM_FILTER CPPFLAGS+=-DBLOOM_FILTER_FILE='\"bloomfilterAll1.raw\"' CPPFLAGS+=-DDEVCPU
    mkdir -p htmlResultsWaterlooBloomAll1/AMDCPU
    echo "Testing full system with bloom filter all ones on CPU"
    OPENCL_FULL "htmlResultsWaterlooBloomAll1/AMDCPU"
fi
python summary.py htmlResultsWaterlooNoBloom
python summary.py htmlResultsWaterlooBloom
python summary.py htmlResultsWaterlooBloomAll0
python summary.py htmlResultsWaterlooBloomAll1
