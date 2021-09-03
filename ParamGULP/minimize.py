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

import collections
import time
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from typing import Callable, Dict, Any
from scipy.optimize import (
    dual_annealing,
    minimize,
    OptimizeResult,
)

# List of the SciPy minimization methods available in ParamGULP
# See SciPy docs for more details related to each method:
#    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
MinimizationMethod = collections.namedtuple("MinimizationMethod", "name description")

# MINIMIZATION_METHODS must be edited to add a new minimization method and the additional kwargs for the method
# must be set in the start_minimization_method function contained in the paramgulp module
MINIMIZATION_METHODS = (
    MinimizationMethod(
        "dual_Nelder-Mead",
        "Minimize with dual annealing using local optimization with Nelder-Mead",
    ),
    MinimizationMethod(
        "dual_Powell",
        "Minimize with dual annealing using local optimization with Powell",
    ),
)


def run_selected_minimization_method(
    method_name: str, objective_function: Callable, x0: list, **kwargs: Dict
) -> OptimizeResult:
    """
    Run a selected minimization method

    :param method_name: Valid SciPy method name
    :param objective_function: Function used with the SciPy minimization method, objective_function
    interfaces the module that handles the GULP files with the minimization method
    :param x0: Initial guess for the adjustable parameters
    :param kwargs: Named args of the SciPy minimization method which are useful to customize as the SciPy method has to work
    :return: SciPy OptimizeResult object. See docs for more details:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    if "dual" in method_name:
        return dual_annealing(objective_function, x0=x0, **kwargs)
    else:
        return minimize(objective_function, x0, method=method_name, **kwargs,)


def abortable_worker(func: Callable, kwds: Dict, timeout: int, method: str) -> Any:
    """
    Call the minimization method and abort it if the timeout is reached

    :param func: start_minimization_method function will be called for each selected minimization method
    :param kwds: Store the arguments to be passed to the start_minimization_method function
    :param timeout: Maximum time to wait a given minimization method running.
    :param method: Method chosen for the parameterization process
    """
    p = ThreadPool(1)
    res = p.apply_async(func, kwds=kwds)
    try:
        # Timeout in seconds for func to complete its execution.
        out = res.get(timeout) if timeout != -1 else res.get(timeout=None)
        return out
    except mp.TimeoutError:
        print(f"Aborting '{method.name}' due to timeout.")
        raise


def run_minimization_methods(func: Callable, **kwds: Dict) -> None:
    """
    Run the minimization methods chosen using the --method parameter in the parallel mode

    :param func: Function responsible for starting the minimization process
    :param kwds: the kwds parameter contains the following parameters:
           option ("all", int, [int]): Store the reference values of the minimization methods chosen by the user
           inp_file_name (str): Name of the .inp file
           output_dir (str): Directory where the files created and used during the parameterization will be saved at
           start_point (int): Number of the start point specified by the user
           n_cycles (int): Maximum number of parametrization cycles
           gulp_path (str): Complete name of the GULP program
           timeout (int): Maximum expected time until each minimization method to be automatically aborted.
           n_cores (int): The number of cores to be used in case of multiple optimizations
                          (--method all or --method A B C). By default, the half of the number of cores is used."
    """
    # Get the timeout and n_cores provided by the user
    timeout = kwds.get("timeout")
    n_cores = kwds.get("n_cores")

    # If --n_cores is defined as 'all', the number of the available cores is used. On the other hand, if
    # --n_cores is empty the half of the number of available cores is used by default.
    n_cores = (
        mp.cpu_count() if n_cores == "all" else ((n_cores or mp.cpu_count() // 2) or 1)
    )

    # Based on the information provided with --method, a list of minimization methods is prepared to be executed
    option = kwds.get("option")
    if option == "all":
        methods = MINIMIZATION_METHODS
    elif isinstance(option, int):
        methods = [MINIMIZATION_METHODS[option - 1]]
    else:
        methods = [MINIMIZATION_METHODS[i - 1] for i in option]

    pool = mp.Pool(processes=int(n_cores), maxtasksperchild=1)
    for i, method in enumerate(methods):
        # Prepare the method argument in kwds to be set in the start_minimization_method function
        kwds["method"] = method

        # The is_new_inp flag prevents the new .inp file created using the parameters contained in POINTS_FILE
        # chosen with --start_point from being created more than once
        kwds["is_new_inp"] = True if i == 0 else False

        pool.apply_async(
            abortable_worker,
            kwds={"func": func, "kwds": kwds, "timeout": timeout, "method": method},
        )
        # The start_minimization_method function is called for the respective minimization method and some files are
        # handled. Sleep is used to prevent concurrent access to the same file"
        time.sleep(2)

    # Wait for all processes to finish
    pool.close()
    pool.join()
