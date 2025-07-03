"""Set up project paths."""
from sklearn.model_selection import train_test_split 
from multiprocessing import Pool
from datetime import datetime
from unidiff import PatchSet
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import subprocess
import hashlib
import inspect
import random
import string
import uuid
import os
import re


def project_dir() -> Path:
     """Get project path."""
    
     return Path(__file__).parent.parent

def storage_dir() -> Path:
    """Get storage path."""
    storage = os.getenv("ENVI")
    if storage:
        return Path(storage) / "storage"
    return Path(__file__).parent.parent / "storage"

def external_dir() -> Path:
    """Get storage external path."""
    path = storage_dir() / "external"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path

def interim_dir() -> Path:
    """Get storage interim path."""
    path = storage_dir() / "interim"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path

def processed_dir() -> Path:
    """Get storage processed path."""
    path = storage_dir() / "processed"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path

def outputs_dir() -> Path:
    """Get output path."""
    path = storage_dir() / "outputs"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path

def cache_dir() -> Path:
    """Get storage cache path."""
    path = storage_dir() / "cache"
    Path(path).mkdir(exist_ok=True, parents=True)
    return path

def get_dir(path) -> Path:
    """Get path, if exists. If not, create it."""
    Path(path).mkdir(exist_ok=True, parents=True)
    return path

def debug(msg, noheader=False, sep="\t"):
    """Print to console with debug information."""
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    if noheader:
        print("\t\x1b[94m{}\x1b[0m".format(msg), end="")
        return
    print(
        '\x1b[40m[{}] File "{}", line {}\x1b[0m\n\t\x1b[94m{}\x1b[0m'.format(
            time, file_name, ln, msg
        )
    )

def gitsha():
    """Get current git commit sha for reproducibility."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode()
    )

def gitmessage():
    """Get current git commit sha for reproducibility."""
    m = subprocess.check_output(["git", "log", "-1", "--format=%s"]).strip().decode()
    return "_".join(m.lower().split())

def subprocess_cmd(command: str, verbose: int = 0, force_shell: bool = False):
    """Run command line process.

    Example:
    subprocess_cmd('echo a; echo b', verbose=1)
    >>> a
    >>> b
    """
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output = process.communicate()
    if verbose > 1:
        debug(output[0].decode())
        debug(output[1].decode())
    return output

def watch_subprocess_cmd(command: str, force_shell: bool = False):
    """Run subprocess and monitor output. Used for debugging purposes."""
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    # Poll process for new output until finished
    noheader = False
    while True:
        nextline = process.stdout.readline()
        if nextline == b"" and process.poll() is not None:
            break
        debug(nextline.decode(), noheader=noheader)
        noheader = True

def genid():
    """Generate random string."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))

def get_run_id(args=None):
    """Generate run ID."""
    if not args:
        ID = datetime.now().strftime("%Y%m%d%H%M_{}".format(gitsha()))
        return ID + "_" + gitmessage()
    ID = datetime.now().strftime(
        "%Y%m%d%H%M_{}_{}".format(
            gitsha(), "_".join([f"{v}" for _, v in vars(args).items()])
        )
    )
    return ID

def hashstr(s):
    """Hash a string."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

def dfmp(df, function, columns=None, ordr=True, workers=6, cs=10, desc="Run: "):
    """Parallel apply function on dataframe.

    Example:
    def asdf(x):
        return x

    dfmp(list(range(10)), asdf, ordr=False, workers=6, cs=1)
    """
    if isinstance(columns, str):
        items = df[columns].tolist()
    elif isinstance(columns, list):
        items = df[columns].to_dict("records")
    elif isinstance(df, pd.DataFrame):
        items = df.to_dict("records")
    elif isinstance(df, list):
        items = df
    else:
        raise ValueError("First argument of dfmp should be pd.DataFrame or list.")

    processed = []
    desc = f"({workers} Workers) {desc}"
    with Pool(processes=workers) as p:
        map_func = getattr(p, "imap" if ordr else "imap_unordered")
        for ret in tqdm(map_func(function, items, cs), total=len(items), desc=desc):
            processed.append(ret)
    return processed

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def train_val_test_split_df(df, idcol, labelcol):
    """Add train/val/test column into dataframe."""
    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=1
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_rat / (test_rat + val_rat), random_state=1
    )
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "val"
        if path in X_test:
            return "test"
    df["label"] = df[idcol].apply(path_to_label)
    return df


def remove_comments(text):
    """Delete comments from code."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)



def gitdiff(old: str, new: str):
    """Git diff between two strings."""
    cachedir = cache_dir()
    oldfile = cachedir / uuid.uuid4().hex
    newfile = cachedir / uuid.uuid4().hex
    with open(oldfile, "w") as f:
        f.write(old)
    with open(newfile, "w") as f:
        f.write(new)
    cmd = " ".join(
        [
            "git",
            "diff",
            "--no-index",
            "--no-prefix",
            f"-U{len(old.splitlines()) + len(new.splitlines())}",
            str(oldfile),
            str(newfile),
        ]
    )
    process = subprocess_cmd(cmd)
    os.remove(oldfile)
    os.remove(newfile)
    return process[0].decode()


def md_lines(patch: str):
    r"""Get modified and deleted lines from Git patch.

    old = "bool asn1_write_GeneralString(struct asn1_data *data, const char *s)\n\
    {\n\
       asn1_push_tag(data, ASN1_GENERAL_STRING);\n\
       asn1_write_LDAPString(data, s);\n\
       asn1_pop_tag(data);\n\
       return !data->has_error;\n\
    }\n\
    \n\
    "

    new = "bool asn1_write_GeneralString(struct asn1_data *data, const char *s)\n\
    {\n\
        if (!asn1_push_tag(data, ASN1_GENERAL_STRING)) return false;\n\
        if (!asn1_write_LDAPString(data, s)) return false;\n\
        return asn1_pop_tag(data);\n\
    }\n\
    \n\
    int test() {\n\
        return 1;\n\
    }\n\
    "

    patch = gitdiff(old, new)
    """
    parsed_patch = PatchSet(patch)
    ret = {"added": [], "removed": [], "diff": ""}
    if len(parsed_patch) == 0:
        return ret
    parsed_file = parsed_patch[0]
    hunks = list(parsed_file)
    assert len(hunks) == 1
    hunk = hunks[0]
    ret["diff"] = str(hunk).split("\n", 1)[1]
    for idx, ad in enumerate([i for i in ret["diff"].splitlines()], start=1):
        if len(ad) > 0:
            ad = ad[0]
            if ad == "+" or ad == "-":
                ret["added" if ad == "+" else "removed"].append(idx)
    return ret

def code2diff(old: str, new: str):
    """Get added and removed lines from old and new string."""
    patch = gitdiff(old, new)
    return md_lines(patch)

def _c2dhelper(item):
    """Given item with func_before, func_after, id, and dataset, save gitdiff."""
    savedir = get_dir(cache_dir() / item["dataset"] / "gitdiff")
    savepath = savedir / f"{item['id']}.git.pkl"
    if os.path.exists(savepath):
        return
    if item["func_before"] == item["func_after"]:
        return
    ret = code2diff(item["func_before"], item["func_after"])
    with open(savepath, "wb") as f:
        pkl.dump(ret, f)
        
def clean_code_snippets(df, code_column):
    def clean_code(code):
        code = re.sub(r'#.*|//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = '\n'.join(line.strip() for line in code.splitlines() if line.strip())
        return code

    df[code_column] = df[code_column].apply(clean_code)

    return df
