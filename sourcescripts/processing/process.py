import os 
import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprocessing import dataset
from dataprocessing import get_dep_add_lines_dataset

def prepared():
    dataset()
    get_dep_add_lines_dataset()
    return 

if __name__ == "__main__":
    prepared()  