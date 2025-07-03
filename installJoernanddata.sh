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

if [[ ! -f "ProjectKB_domain_csv" ]]; then
    gdown https://drive.google.com/uc\?id\=1W3Truwd_kEAGYwBgHd6hRLELAeyQRBwT
    unzip ProjectKB_domain_csv.zip
    rm ProjectKB_domain_csv.zip
else
    echo "Already downloaded dataset data"
fi


# https://drive.google.com/file/d/1W3Truwd_kEAGYwBgHd6hRLELAeyQRBwT/view?usp=sharing