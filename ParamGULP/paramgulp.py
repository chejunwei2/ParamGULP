#!/usr/bin/python3

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


import argparse
import functools
import itertools
import os
import shutil
import sys
import time
from typing import Tuple, Dict

from scipy.optimize import OptimizeResult

from common import (
    split2float,
    POINTS_FILE,
    MAX_OBJECTIVE_VALUE,
    OUTPUT_DIR_NAME,
    ANALYSIS_FILE,
    CYCLES_NUMBER,
    DEFAULT_TIMEOUT,
    SKIP_GRAD,
    GULP_TIME_OUT,
    GULP_PATH,
    TOL_PARAMS,
    TOL_CYCLES,
    TOL_IGNORE,
)
from gulp import ParametricGulp
from minimize import (
    MINIMIZATION_METHODS,
    run_selected_minimization_method,
    run_minimization_methods,
)


class CyclesNumberReached(Exception):
    pass


def get_params_out_of_range(
    params: list, lower_params: list, upper_params: list
) -> list:
    """
    Check if any parameter specified by the user is out of the range that was defined

    :param params: List of parameters read from the .inp file
    :param lower_params: List of lower bounds provided by the user in the .inp file
    :param upper_params: List of upper bounds provided by the user in the .inp file

    :return: List of parameters out of the defined range
    """
    params_out = [
        i
        for i in range(len(lower_params))
        if params[i] < lower_params[i] or params[i] > upper_params[i]
    ]

    return params_out


def params_from_points_file(output_dir: str, start_point: int) -> list:
    """
    Obtain the set of parameters from the POINTS_FILE file when the user specifies a start point different from 0
    to create a new .inp file

    :param output_dir: Directory where the outputs files are saved
    :param start_point: Start point specified by the user to start the parametrization process
    :return: List of parameters obtained from the points.txt file
    """

    base_dir = get_base_dir(output_dir)
    points_file = os.path.join(base_dir, POINTS_FILE)

    params = []
    try:
        with open(points_file) as file:
            for str_list in map(str.split, file):
                if len(str_list) == 1 and int(str_list[0]) == start_point:
                    for line_aux in map(str.strip, file):
                        if not line_aux or "Objective Function" in line_aux:
                            break

                        params.append(split2float(line_aux, 1))
                    break
    except OSError:
        return []

    return params


def objective_function(
    x: list,
    method_name: str,
    gulp: ParametricGulp,
    lower_params: list,
    upper_params: list,
    n_cycles: int,
    tol_params: float,
    tol_cycles: int,
    tol_ignore: int,
) -> float:
    """
    Create the interface between the ParametricGulp object and the scipy minimization algorithm

    :param x: List of parameters that is provided by the minimization algorithm and will be used in the
    current parametrization cycle
    :param method_name: Minimization algorithm to be used
    :param gulp: ParametricGulp object which handles the GULP files
    :param lower_params: List of lower bounds provided by the user through the .inp file
    :param upper_params: List of upper bounds provided by the user through the .inp file
    :param n_cycles: Maximum number of parametrization cycles
    :param tol_params: XXXXXXXXXXXXXXXXXXXXXXXXXX
    :param tol_cycles: XXXXXXXXXXXXXXXXXXXXXXXXXX
    :param tol_ignore: XXXXXXXXXXXXXXXXXXXXXXXXXX

    :return: objective function calculated by using the GULP output file (.gout) considering the
    geometric parameters (a, b, c, and V)
    """

    if get_params_out_of_range(x, lower_params, upper_params):
        return MAX_OBJECTIVE_VALUE

    x_rounded = tuple(round(v, 4) for v in x)

    if x_rounded in gulp.params_cache:
        return gulp.params_cache[x_rounded]

    gulp.inc_point_number()

    current_point_min = gulp.point_number_min
    gulp.run(x)

    print(
        ">>>>",
        method_name,
        f"point_number={gulp.point_number}   obj_function={gulp.objective_function:.6e}\n{'-' * 45}",
    )

    if gulp.point_number >= n_cycles:
        message = f">>>> {method_name}: The maximum number of cycles (n_cycles={n_cycles}) was reached."
        print(f"{message}\n{'-' * 45}")
        raise CyclesNumberReached(message)

    gulp.params_cache[x_rounded] = gulp.objective_function

    if not tol_ignore and gulp.tol_cycles_reached(tol_cycles):
        print(
            f">>>> {method_name}: As the objective function could not be reduced after running "
            f"{tol_cycles} parameterization cycles, the parameterization was terminated."
        )
        raise

    if (
        current_point_min != gulp.point_number_min
        and gulp.point_number > 1
        and not tol_ignore
    ):
        PARAMS_CACHE_LIST = list(gulp.params_cache.keys())

        params_current = list(PARAMS_CACHE_LIST[current_point_min - 1])
        params_new = list(PARAMS_CACHE_LIST[gulp.point_number_min - 1])

        diff = [
            round(abs(value - params_new[j]), 4)
            for j, value in enumerate(params_current)
        ]

        if len([d for d in diff if d <= tol_params]) == len(diff):
            print(
                f">>>> {method_name}: The maximum tolerance on parameters of {tol_params} was reached."
            )
            raise

    return gulp.objective_function


def get_base_dir(output_dir: str) -> str:
    """
    Return the base directory from the output_dir

    :param output_dir: Output directory where the files created during the parameterization will be saved at
    :return: base directory (equivalent to the .inp file directory)
    """

    return output_dir.replace(OUTPUT_DIR_NAME, "")


def save_previous_minimized_data(output_dir: str) -> None:
    """
    Save the points.txt and analysis.txt files that were created in the last parameterization at a folder renamed
    with a given index.

    :param output_dir: Directory where the files created during the parameterization will be saved at
    """
    base_dir = get_base_dir(output_dir)

    dirs = os.listdir(base_dir)
    # The default content of OUTPUT_DIR_NAME is "minimize_output"
    dirs = [d for d in dirs if OUTPUT_DIR_NAME in d]

    # Extract the index of the end of the folder name
    idx = [int(d.split("_")[-1]) for d in dirs if d.split("_")[-1].isdigit()]

    if idx:
        # Increment 1 to the largest index for renaming the last output directory
        dir_ = f"{output_dir}_{max(idx)+1}"
    elif os.path.exists(output_dir):
        # If the output directory exists and it was still not renamed, the first index will be added
        dir_ = f"{output_dir}_1"

    try:
        os.rename(output_dir, dir_)
    except:
        if os.path.exists(output_dir):
            print(
                f"Please close all the files located at {output_dir} because the folder will be renamed."
            )
            sys.exit()

    os.makedirs(output_dir, 0o755, True)


def configure_to_startpoint(
    gulp: ParametricGulp,
    inp_file_name: str,
    output_dir: str,
    start_point: int,
    is_new_inp: bool,
) -> None:
    """
    Save a new .inp file containing the set of parameters read from points.txt
    when the user specified a valid start point

    :param gulp: ParametricGulp object
    :param inp_file_name: .inp file name
    :param output_dir: Directory where the files created during the parameterization will be saved at
    :param start_point: Number of the start point specified by the user
    :param is_new_inp: Check if the new .inp file was created from POINTS_FILE
    """
    # Extract the list of parameters from the
    parameters_starting = params_from_points_file(output_dir, start_point)

    base_dir = get_base_dir(output_dir)
    if not parameters_starting:
        print(
            f"The selected point has no parameters or {os.path.join(base_dir, POINTS_FILE)} does not exist."
        )
        raise

    if len(gulp.params) != len(parameters_starting):
        print("It is not possible to start from the selected parameters.")
        raise
    else:
        if is_new_inp:
            shutil.copy(inp_file_name, inp_file_name.replace(".inp", ".inp_old"))
            gulp.export_gulp_input_with_starting(parameters_starting)


def prepare_output_dir_for_method(
    method_name: str, output_dir: str, inp_file_name: str
) -> Tuple[str, str]:
    """
    Prepare the output directory where the results will be saved at

    :param method_name: Name of the minimization algorithm that will be used in the parameterization procedure
    :param output_dir: Directory where the files will be saved at
    :param inp_file_name: Name of the .inp file
    :return: Output directory and the full .gin file name
    """
    output_path = os.path.join(
        os.path.dirname(inp_file_name), output_dir, method_name.lower()
    )
    os.makedirs(output_path, exist_ok=True)

    base_name = os.path.basename(inp_file_name).replace(".inp", ".gin")

    # .gin is the file effectively called by GULP in the parameterization process.
    input_gin_file = os.path.join(output_path, base_name)

    return output_path, input_gin_file


def start_minimization_method(**kwds: Dict) -> OptimizeResult:
    """
    Prepare and start the parametrization procedure

    :param kwds: the kwds argument contains the following parameters:
           method (MinimizationMethod): Name of the minimization algorithm
           inp_file_name (str): Name of the .inp file
           output_dir (str): Directory where the files created and used during the parametrization will be saved at
           start_point (int): Number of the start point specified by the user
           n_cycles (int): Maximum number of parametrization cycles
           gulp_path (str): GULP program localization
           skip_grad (float): If the gradient calculated by GULP is larger than --skip_grad,
                      the evaluated parameters will not be considered.
           gulp_timeout (int): Maximum time (in seconds) to wait GULP to finish the calculations.

    :return: Call the minimize function to start the parametrization
    """
    # Extract the values from the arguments provided by the user.
    method = kwds.get("method")
    inp_file_name = kwds.get("inp_file_name")
    output_dir = kwds.get("output_dir")
    start_point = kwds.get("start_point")
    n_cycles = kwds.get("n_cycles")
    gulp_path = kwds.get("gulp_path")
    skip_grad = kwds.get("skip_grad")
    gulp_timeout = kwds.get("gulp_timeout")
    is_new_inp = kwds.get("is_new_inp")
    obj_func_type = kwds.get("obj_func_type")
    tol_params = kwds.get("tol_params")
    tol_cycles = kwds.get("tol_cycles")
    tol_ignore = kwds.get("tol_ignore")

    # Create an object of the class that handles the GULP input and output files.
    gulp = ParametricGulp(inp_file_name, gulp_path)

    lower_params, upper_params = gulp.limits

    # If any parameter is out of range the execution of ParamGULP is broken
    params_out = get_params_out_of_range(gulp.params, lower_params, upper_params)
    if params_out:
        print(f"There are starting parameters out of range: \t {params_out}")
        raise

    gulp.output_dir, gulp.gin_file_name = prepare_output_dir_for_method(
        method.name, output_dir, inp_file_name
    )
    gulp.skip_grad = skip_grad
    gulp.gulp_timeout = gulp_timeout
    gulp.obj_func_type = obj_func_type

    # Create a new .inp file from the set of parameters presents in POINT_FILE if --start_point is different from 0.
    if start_point != 0:
        configure_to_startpoint(
            gulp, inp_file_name, output_dir, start_point, is_new_inp
        )

    # Run a GULP calculation considering the .inp file (initial guess parameters)
    # to check if the file is valid for GULP.
    if not gulp.is_gulp_input_valid():
        print(
            f"The GULP .inp file provided is not valid. Please take a look at the file\n"
            f"{gulp.gin_file_name.replace('.gin', '.gout')} to know why GULP terminated with an error."
        )
        raise

    # Assign auxiliary parameters to the objective function which serves as an interface for the minimization methods.
    func = functools.partial(
        objective_function,
        method_name=method.name,
        gulp=gulp,
        lower_params=lower_params,
        upper_params=upper_params,
        n_cycles=n_cycles,
        tol_params=tol_params,
        tol_cycles=tol_cycles,
        tol_ignore=tol_ignore,
    )

    kwargs = {}
    # A traditional Generalized Simulated Annealing will be performed with no local search strategy applied.
    if method.name == "dual":
        kwargs = {
            "bounds": list(zip(lower_params, upper_params)),
            "no_local_search": True,
            "seed": 5,
            "maxfun": MAX_OBJECTIVE_VALUE,
        }
    elif method.name == "dual_Nelder-Mead":
        kwargs = {
            "bounds": list(zip(lower_params, upper_params)),
            "no_local_search": False,
            "local_search_options": {"method": "Nelder-Mead",},
            "seed": 5,
            "maxfun": MAX_OBJECTIVE_VALUE,
        }
    elif method.name == "dual_Powell":
        kwargs = {
            "bounds": list(zip(lower_params, upper_params)),
            "no_local_search": False,
            "local_search_options": {"method": "Powell"},
            "seed": 5,
            "maxfun": MAX_OBJECTIVE_VALUE,
        }

    return run_selected_minimization_method(method.name, func, gulp.params, **kwargs)


def handle_analysis_files(output_dir: str) -> None:
    """
    Create a file named output.txt containing the best results obtained when the --method=all is used.

    :param output_dir: Directory where the files created and used during the parametrization will be saved at
    """

    dirs_list = os.listdir(output_dir)
    analysis_data_all = []
    for dir_ in dirs_list:
        # As the results are stored in folder with the same name as the respective minimization method,
        # it is necessary to check if the dir_ variable is contained in the list of minimization methods
        if not any(dir_ for m in MINIMIZATION_METHODS if dir_ in m.name.lower()):
            continue

        analysis_file_name = os.path.join(output_dir, dir_, ANALYSIS_FILE)

        analysis_data = ()
        if os.path.exists(analysis_file_name):
            with open(analysis_file_name) as input_file:
                for line in input_file:
                    if "POINT NUMBER" in line:
                        next(input_file)
                        buffer = [f"{line}\n"]

                        for line_aux in input_file:
                            if "*********" in line_aux:
                                break
                            if "Objective Function" in line_aux:
                                objective_fun = split2float(line_aux, 3)
                            buffer.append(line_aux)
                        else:
                            # Save the results for the last POINT NUMBER and break the file reading
                            analysis_data = (objective_fun, "".join(buffer))
                            break

        if analysis_data:
            # Append the result for the lowest objective function for each method ([1])
            # together the corresponding Fobj ([0])
            analysis_data_all.append(
                (analysis_data[0], f"{dir_}\n\n{analysis_data[1]}")
            )

    # Sort the all results in ascending order of objective function (Fobj) in order to facilitate to
    # find the best result in the output.txt file
    analysis_data_all.sort()

    # Save output.txt containing the best results for each method
    if analysis_data_all:
        output_file_name = os.path.join(output_dir, "output.txt")
        with open(output_file_name, "w") as output_file:
            for analysis_data in analysis_data_all:
                output_file.write(f"\n{'*' * 45}\n{analysis_data[1]}\n")


def run_analysis_gout(file_name):
    try:
        gulp = ParametricGulp(file_name, gulp_path="", is_analysis_gout=True)
        gulp.is_gulp_input_valid(is_exec_gulp=False)
        gulp._calc_objective_function(is_exec_gulp=False)
        gulp._append_analysis_file()
    except:
        print("Problem during analysis of the .gout file.")
        pass


def main(**kwargs: Dict) -> None:
    output_dir = kwargs.get("output_dir")

    save_previous_minimized_data(output_dir)

    run_minimization_methods(
        start_minimization_method, **kwargs,
    )

    # If more than one minimization method to be carried out,
    # the output.txt file containing the best results for each method will be created.
    if not isinstance(option, int):
        handle_analysis_files(output_dir)


if __name__ == "__main__":
    """
    Obtain the arguments provided by user using the terminal
    """
    parser = argparse.ArgumentParser(
        description="ParamGULP is a code designed to obtain any interatomic potential parameter implemented "
        "in the General Utility Lattice Program (GULP) using many SciPy minimization methods"
    )

    options = map(str, range(0, len(MINIMIZATION_METHODS) + 1))
    options = tuple(itertools.chain(("all",), options))
    options_joined = (
        "0: an analysis is performed from the desired .gin and .gout files, "
    )
    options_joined += ", ".join(
        f"{i}:{m.name}" for i, m in enumerate(MINIMIZATION_METHODS, start=1)
    )

    parser.add_argument(
        "--method",
        required=True,
        nargs="+",
        choices=options,
        help=f"Desired minimization methods: all, {options_joined}. ",
    )

    parser.add_argument(
        "--input_file", required=True, type=str, help="GULP input file name",
    )

    parser.add_argument(
        "--gulp_path",
        type=str,
        default=GULP_PATH,
        help="GULP executable location(Windows default: C:/gulp/gulp; UNIX default: ~/gulp/Src/gulp)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help=f"Output path to save the created files. Directory named '{OUTPUT_DIR_NAME}' "
        f" will be created to save the files created by this program. "
        f"The default name is --input_file dir",
    )

    parser.add_argument(
        "--start_point",
        type=int,
        default=0,
        help="Start point from which the parametrization will be started",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Maximum expected time (in seconds) until each optimization method to be automatically aborted. "
        "By default, the optimization method will be aborted when --n_cycles is reached.",
    )

    parser.add_argument(
        "--n_cycles",
        type=int,
        default=CYCLES_NUMBER,
        help="Maximum number of parametrization cycles. Depending on certain criteria of the minimization "
        f"method, it is possible that this number can not be reached. Default value is {CYCLES_NUMBER}",
    )

    parser.add_argument(
        "--n_cores",
        type=str,
        default=None,
        help="The number of cores to be used in case of multiple optimizations (--method all). "
        "To use all available cores, set this parameter as 'all'. By default, the half of the number of cores is used",
    )

    parser.add_argument(
        "--gulp_timeout",
        type=int,
        default=GULP_TIME_OUT,
        help="Maximum time (in seconds) to wait GULP to finish the calculations. "
        "Default value will be equal to 5x the spent time in the first parameterization cycle",
    )

    parser.add_argument(
        "--skip_grad",
        type=float,
        default=SKIP_GRAD,
        help=f"If the gradient calculated by GULP is larger than --skip_grad, the evaluated "
        f"parameters will not be considered. Default value is {SKIP_GRAD}",
    )

    parser.add_argument(
        "--obj_func",
        type=int,
        default=0,
        help="Default value of --obj_func is 0 and the objective function based on the sum of squares is used. "
        "Otherwise, the objective function is calculated using a sum of percentage relative errors",
    )

    parser.add_argument(
        "--tol_params",
        type=float,
        default=TOL_PARAMS,
        help=f"Tolerance for variations in parameters. Default value of --tol_params is {TOL_PARAMS}",
    )

    parser.add_argument(
        "--tol_cycles",
        type=int,
        default=TOL_CYCLES,
        help=f"The parameterization process will be interrupted if after evaluating {TOL_CYCLES} "
        f"parameterization cycles the objective function is not reduced",
    )

    parser.add_argument(
        "--tol_ignore",
        type=int,
        default=TOL_IGNORE,
        help=f"If different from 0, the tolerance criteria will not be applied. "
        f"Default value of --tol_ignore is {TOL_PARAMS}",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if not args.output_dir:
        output_dir = os.path.join(os.path.dirname(args.input_file), OUTPUT_DIR_NAME)
    else:
        output_dir = os.path.expanduser(args.output_dir)
        output_dir = os.path.join(output_dir, OUTPUT_DIR_NAME)

    if args.method[0] == "0":
        run_analysis_gout(args.input_file)
        sys.exit(0)

    gulp_path = args.gulp_path if args.gulp_path else GULP_PATH
    if not os.path.exists(gulp_path):
        print(
            f"{gulp_path} could not be found. Please provide a valid value for --gulp_path."
        )
        sys.exit(0)

    inp_file_name = os.path.abspath(os.path.expanduser(args.input_file))
    if not os.path.exists(inp_file_name):
        print(
            f"{inp_file_name} could not be found. Please provide a valid value for --input_file."
        )
        sys.exit(0)

    _, extension = os.path.splitext(inp_file_name)
    if not extension or extension != ".inp":
        print(
            f"The extension of the {inp_file_name} file must be .inp. \nPlease check your command line."
        )
        sys.exit(0)

    if len(args.method) == 1:
        option = "all" if args.method[0] == "all" else int(args.method[0])
    else:
        option = tuple(map(int, args.method))

    n_cores = args.n_cores
    if n_cores:
        n_cores = int(args.n_cores) if args.n_cores != "all" else "all"

    t0 = time.time()

    output_dir = os.path.join(os.path.dirname(inp_file_name), output_dir)

    main(
        option=option,
        inp_file_name=inp_file_name,
        output_dir=output_dir,
        gulp_path=gulp_path,
        start_point=args.start_point,
        timeout=args.timeout,
        n_cycles=args.n_cycles,
        n_cores=args.n_cores,
        skip_grad=args.skip_grad,
        gulp_timeout=args.gulp_timeout,
        obj_func_type=args.obj_func,
        tol_params=args.tol_params,
        tol_cycles=args.tol_cycles,
        tol_ignore=args.tol_ignore,
    )

    t1 = time.time()
    print(f"elapsed time={t1 - t0:.2f}s")
