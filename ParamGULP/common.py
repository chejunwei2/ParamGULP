"""
Copyright (c) 2020-2021 by 
   José Diogo Lisboa Dutra <jdiogo@ufs.br>
   Thiago Dias Bispo <thiagodiasbispo@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import os
import subprocess
import time
import platform
from typing import Union, Optional

POINTS_FILE = "points.txt"  # File where the point provided by SciPy minimization algorithm are stored
ANALYSIS_FILE = "analysis.txt"  # File that stores the best results
OUTPUT_DIR_NAME = "minimize_output"  # Default value for --output_dir
COMPOUNDS_FILE = "compounds.txt"  # File that stores the weight values for all compounds
MAX_OBJECTIVE_VALUE = 1e30  # Maximum value for the objective function
CYCLES_NUMBER = 1000  # Default value for --n_cores
X_ELAPSED_GULP_TIME = 5
GULP_TIME_OUT = (
    -1
)  # Default value for --gulp_timeout (in seconds) will be equal to ELAPSED_GULP_TIME x the spent time in the first
# parameterization cycle

DEFAULT_TIMEOUT = (
    -1
)  # If a value is not provided for --timeout, the optimization method will be aborted when
# the number of cycles is reached
SKIP_GRAD = 0.1  # Default value for --skip_grad
AU2EV = 27.211324570273  # Factor to convert a value from au to eV (energy observable)
TOL_PARAMS = 0.001  # Tolerance for variations in parameters
TOL_CYCLES = 200  # The parameterization process will be interrupted if after evaluating 200 parameterization cycles
# the objective function is not reduced,
TOL_IGNORE = 0  # If different from 0, the tolerance criteria will not be considered

# The GULP_PATH variable is changed by the --gulp_path parameter
if platform.system() == "Windows":
    GULP_PATH = "C:/gulp/gulp.exe"
else:
    GULP_PATH = os.path.expanduser("~/gulp/Src/gulp")


def iter_find(source: str, flag: str) -> list:
    """
    Find the indexes of the source parameter that contains the provided flag. This function is important
    to detect the 'parameter_lowerbound_upperbound' syntax contained in the .inp file

    :param source: Source string
    :param flag: String to be searched
    :return: List with the indexes or an empty list
    """
    return [i for i in range(len(source)) if source[i] == flag]


def split2float(source: str, idx: int, default: float = 0.0) -> float:
    """
    Helper function to split a source and to get its idx value, besides converting it to float.
    If the operations fail, the default value is returned

    :param source: Source string
    :param idx: Desired index
    :param default: Default value in case of failure
    :return: The float value obtained from the split source
    """
    try:
        return float(source.split()[idx])
    except (IndexError, ValueError):
        raise


def exec_command(
    args: Union[list, tuple], path: str, gulp_timeout: Optional[int] = None
) -> None:
    """
    Execute the GULP program using the .gin and .gout files as input and output, respectively

    :param args: Details of the command to be carried out
    :param path: Directory where is the .gin file.
    :param gulp_timeout: Chosen timeout to stop the calculation with GULP, the default value is 15 seconds
    """

    process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, cwd=path)
    start = time.time()
    try:
        process.communicate(timeout=gulp_timeout)
    except subprocess.TimeoutExpired:
        process.kill()
    except KeyboardInterrupt as e:
        raise e
    finally:
        end = time.time()
        elapsed_time = end - start
        print(f"Gulp terminated in {elapsed_time:.4f}s")

        return elapsed_time
