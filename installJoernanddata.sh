if [[ -d sourcescripts/storage/external ]]; then
    echo "storage exists, starting download"
else
    mkdir --parents sourcescripts/storage/external
fi

cd sourcescripts/storage/external

# https://github.com/joernio/joern/releases/download/v4.0.398/joern-cli.zip
if [[ ! -d joern-cli ]]; then
    wget https://github.com/joernio/joern/releases/download/v2.0.331/joern-cli.zip
    unzip joern-cli.zip
    rm joern-cli.zip
else
    echo "Already downloaded Joern"
fi

if [[ ! -f "ProjectKB_domain_csv" ]]; then
    gdown https://drive.google.com/uc\?id\=1W3Truwd_kEAGYwBgHd6hRLELAeyQRBwT
    unzip ProjectKB_domain_csv.zip
    rm ProjectKB_domain_csv.zip
else
    echo "Already downloaded dataset data"
fi


# https://drive.google.com/file/d/1W3Truwd_kEAGYwBgHd6hRLELAeyQRBwT/view?usp=sharing

cd .. 

cd processed

# https://drive.google.com/file/d/1wcoXzQEWxCDWmhC4mtCB4nHIGcIbeoNW/view?usp=sharing

if [[ ! -f "PJkb.zip" ]]; then
    gdown https://drive.google.com/uc\?id\=1wcoXzQEWxCDWmhC4mtCB4nHIGcIbeoNW
    unzip PJkb.zip
    rm PJkb.zip
else
    echo "Already downloaded dataset data"
fi

cd ..

mkdir cache -p

cd cache

# https://drive.google.com/file/d/1NLxOqBHrU2H3oswqBvjL-fn8Ipe2o7a0/view?usp=sharing
if [[ ! -f "embedmodel" ]]; then
    gdown https://drive.google.com/uc?id=1NLxOqBHrU2H3oswqBvjL-fn8Ipe2o7a0
    unzip embedmodel.zip
    rm embedmodel.zip
else
    echo "Already downloaded finetuned embedding model"
fi


