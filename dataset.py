import os
import numpy as np
import pickle
from wordtable import *


def prepare_data(arg):
    """"build the word table """
    word_table = WordTable();