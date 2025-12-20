if [[ -d sourcescripts/storage/external ]]; then
    echo "storage exists, starting download"
else
    mkdir --parents sourcescripts/storage/external
fi

cd sourcescripts/storage/external

if [[ ! -d joern-cli ]]; then
    wget https://github.com/joernio/joern/releases/download/v2.0.331/joern-cli.zip
    unzip joern-cli.zip
    rm joern-cli.zip
else
    echo "Already downloaded Joern"
fi


#------ Bigvul data

if [[ ! -f "MSR_data_cleaned.csv" ]]; then
    gdown https://drive.google.com/uc\?id\=1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X
    unzip MSR_data_cleaned.zip
    rm MSR_data_cleaned.zip
else
    echo "Already downloaded bigvul data"
fi

cd .. 

mkdir processed -p

cd processed

cd ..

mkdir cache -p

cd cache

if [[ ! -f "embedmodel" ]]; then
    gdown https://drive.google.com/uc?id=1NLxOqBHrU2H3oswqBvjL-fn8Ipe2o7a0
    unzip embedmodel.zip
    rm embedmodel.zip
else
    echo "Already downloaded finetuned embedding model"
fi


