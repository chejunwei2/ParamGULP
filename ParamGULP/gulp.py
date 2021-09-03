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

import itertools
import os
from typing import Tuple, TextIO
from common import (
    split2float,
    iter_find,
    exec_command,
    POINTS_FILE,
    COMPOUNDS_FILE,
    MAX_OBJECTIVE_VALUE,
    ANALYSIS_FILE,
    GULP_PATH,
    GULP_TIME_OUT,
    SKIP_GRAD,
    AU2EV,
    X_ELAPSED_GULP_TIME,
)
import collections

# Data type created to store the reference observables which are found in the .inp file, and
# the calculated observables which are read from the .gout file
ObservablesData = collections.namedtuple(
    "ObservablesData", "name observable indexes value"
)

# Data type that stores the reference and calculated data. These data are used in the calculations of the
# objective function and are also saved in the analysis.txt file for the best objective functions
ObjectiveFunctionData = collections.namedtuple(
    "ObjectiveFunctionData",
    "name observable indexes ref calc squared_error relative_error",
)


class ParametricGulp:
    """
    Class designed to handle the GULP input and output files,
    besides running the GULP executable provided by the gulp_path parameter
    """

    def __init__(self, file_name: str, gulp_path: str, is_analysis_gout: bool = False):
        self.gulp_path = gulp_path or GULP_PATH
        self._inp_file_name = file_name
        self._data_list = []
        self._compounds_name = []

        # Special condition that occurs if the initial guess parameters
        # provide an unsatisfactory GULP calculation.
        self.get_detected_error = ""
        self._special_condition_disabled = False

        # Default values for the variables
        self._objective_function_min = MAX_OBJECTIVE_VALUE
        self._objective_function = MAX_OBJECTIVE_VALUE
        self._point_number = 0
        self._point_number_min = 0
        self._output_dir = None
        self._gin_file_name = None
        self._skip_grad = SKIP_GRAD
        self._gulp_timeout = GULP_TIME_OUT

        self._last_objective_function = MAX_OBJECTIVE_VALUE

        # 0 -> error sum of squared differences; otherwise -> percentage relative error
        self.obj_func_type = 0
        self.is_analysis_gout = is_analysis_gout
        if self.is_analysis_gout:
            self._gin_file_name = self._inp_file_name
            self._output_dir = os.path.dirname(self._gin_file_name)
            self._inp_file_name = os.path.join(self._output_dir, self._inp_file_name)

        self._read_inp_gulp()

        # Get observables to be take into account in the parameterization
        self._observables_ref = []
        self._get_observables()

        if self.is_analysis_gout:
            return

        # Get the adjustable parameters from the .inp file when the
        # 'parameter_lowerbound_upperbound' syntax is found
        self._get_params_gulp()

        self.params_cache = {}

    @property
    def objective_function(self) -> float:
        return self._objective_function

    @property
    def last_objective_function(self) -> float:
        return self._last_objective_function

    @property
    def objective_function_min(self) -> float:
        return self._objective_function_min

    def inc_point_number(self) -> None:
        self._point_number += 1

    @property
    def point_number(self) -> int:
        return self._point_number

    @property
    def point_number_min(self) -> int:
        return self._point_number_min

    @property
    def inp_file_name(self) -> str:
        return self._inp_file_name

    @inp_file_name.setter
    def inp_file_name(self, name: str) -> None:
        self._inp_file_name = name

    @property
    def gout_file_name(self) -> str:
        return self.gin_file_name.replace(".gin", ".gout")

    @property
    def output_dir(self) -> str:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, dir_) -> None:
        self._output_dir = dir_

    @property
    def skip_grad(self) -> float:
        return self._skip_grad

    @skip_grad.setter
    def skip_grad(self, skip_grad_) -> None:
        self._skip_grad = skip_grad_

    @property
    def obj_func_type(self) -> int:
        return self._obj_func_type

    @obj_func_type.setter
    def obj_func_type(self, obj_func_type_) -> None:
        self._obj_func_type = obj_func_type_

    @property
    def gulp_timeout(self) -> int:
        return self._gulp_timeout

    @gulp_timeout.setter
    def gulp_timeout(self, gulp_timeout) -> None:
        self._gulp_timeout = gulp_timeout

    @property
    def gin_file_name(self) -> str:
        return self._gin_file_name

    @gin_file_name.setter
    def gin_file_name(self, gin_file_name) -> None:
        self._gin_file_name = gin_file_name

    @property
    def points_file(self) -> str:
        return self._get_path_output_file(POINTS_FILE)

    @property
    def analysis_file(self) -> str:
        return self._get_path_output_file(ANALYSIS_FILE)

    @property
    def limits(self) -> Tuple[list, list]:
        return self.lower_params, self.upper_params

    def tol_cycles_reached(self, tol_cycles):
        return (
            self.point_number - self.point_number_min >= tol_cycles
            and self.objective_function >= self.objective_function_min
        )

    def handle_compounds_file(self):
        """

        :return:
        """
        base_dir = os.path.dirname(self.inp_file_name)

        # Handle the compounds.txt file, which will not be created if it already exists
        self._compounds_file_name = os.path.join(base_dir, COMPOUNDS_FILE)
        self._create_compound_weight_file()
        self._fill_compound_weight_list()

    def is_gulp_input_valid(self, is_exec_gulp=True) -> bool:
        """
        Check if the .inp file provided by the user is valid. Such checking is done before
        the first parameterization cycle

        :return: True if the first calculation with GULP considering the .inp file is successfully completed and
        False otherwise
        """
        if is_exec_gulp:
            self._export_next_gulp_input(self.params)

            # Check if the provided GULP .inp is valid, a larger gulp_timeout is used (2 min.)
            elapsed_time = self._exec_gulp_command(120)
            with open(self.gout_file_name) as file:
                if not "Job Finished" in file.read():
                    return False

            if self._gulp_timeout == -1:
                self._gulp_timeout = elapsed_time * X_ELAPSED_GULP_TIME
                print(
                    f"Timeout for the gulp calculation was set as {self._gulp_timeout:.4f}s, \n"
                    f"which is {X_ELAPSED_GULP_TIME} times the time spent in the calculation for the initial set of parameters."
                )

        with open(self.gout_file_name) as file:
            # Check if _observables_ref already contains some observables already read in the .inp file
            observable_ref_aux = []
            if self._observables_ref:
                observable_ref_aux, self._observables_ref = self._observables_ref, []

            is_compare_kw = False
            angles = ["alpha", "beta", "gamma"]
            data_temp = []
            name = ""
            try:
                for line in file:
                    if "Output for configuration" in line:
                        data_temp.sort(key=lambda l: l.indexes)
                        for o in observable_ref_aux:
                            if o.name == name:
                                data_temp.append(o)

                        data_temp.sort(key=lambda l: l.observable)

                        name = line.split()[6]

                        self._observables_ref.extend(data_temp)
                        data_temp = []

                    if "Comparison of initial and final structures" in line:
                        is_compare_kw = True

                        [next(file) for _ in range(4)]
                        for aux_line in file:
                            if "----------------" in aux_line:
                                break

                            str_list = aux_line.split()
                            index = "_".join(str_list[:-5])
                            if index in angles:
                                continue

                            value = float(str_list[-5])

                            data_temp.append(
                                ObservablesData(name, "cell", index, value)
                            )
            except:
                if not is_compare_kw:
                    print(
                        f"The lattice parameters could not be found at the .gout file. Probably the "
                        f"'compare' and/or 'name' keywords were not added to the .inp file."
                    )

                return False

            data_temp.sort(key=lambda l: l.indexes)
            for o in observable_ref_aux:
                if o.name == name:
                    data_temp.append(o)

            data_temp.sort(key=lambda l: l.observable)

            self._observables_ref.extend(data_temp)

        if not self._observables_ref:
            return False

        self.handle_compounds_file()

        return True

    def run(self, params: list) -> None:
        """
        Run one parameterization cycle considering the list of parameters (new set of parameters)
        provided by the SciPy minimization method. A new GULP input file (.gin) is created containing the new set of
        parameters and the correspondent GULP output file (.gout) is created when GULP is carried out

        :param params: List of parameters provided by the SciPy minimization method
        """
        self._export_next_gulp_input(params)
        self._calc_objective_function()
        self._setup_if_new_min_objective_function()
        self._append_points_file(params)

    def export_gulp_input_with_starting(self, params: list) -> None:
        """
        Export a GULP .inp file containing the set of parameters read from the points.txt file and
        extract the adjustable parameters from the created .inp file

        :param params: List of parameters read from the points.txt file
        """
        buffer_aux = list(self.buffer_inp_file)

        for i, j in itertools.product(range(len(buffer_aux)), range(len(params))):
            buffer_aux[i] = buffer_aux[i].replace(
                f"PARAM_{j} ",
                f"{params[j]:.4f}_{self.lower_params[j]:.2f}_{self.upper_params[j]:.2f}",
            )

        with open(self.inp_file_name, "w") as inp_file:
            inp_file.writelines(buffer_aux)

        self._read_inp_gulp()
        self._get_params_gulp()

    def _create_compound_weight_file(self) -> None:
        """
        Create the compounds.txt file, which contains the default weight values (1.0)
        for all compounds detected in the .inp file. The compounds are found from the 'name' flag
        """

        def get_weight_squared_obj_func(observable):
            if "cell" in observable:
                if "_x" in observable or "_y" in observable or "_z" in observable:
                    return "cell_xyz 10000.0  "
                else:
                    return "cell_abc 1000.0  "

            if "elast" in observable:
                return "elastic 0.01  "

            if "hfdlc" in observable:
                return "hfdlc 1.0  "

            if "sdlc" in observable:
                return "sdlc 1.0  "

            return f"{observable} 1.0 "

        def get_weight_relative_obj_func(observable):
            if "cell" in observable:
                if "_x" in observable or "_y" in observable or "_z" in observable:
                    observable = "cell_xyz"
                else:
                    observable = "cell_abc"

            return f"{observable} 1.0  "

        if os.path.exists(self._compounds_file_name):
            with open(self._compounds_file_name) as compounds_file:
                compounds_name = [l.split()[0] for l in compounds_file]
                # Compare if the compounds contained in the existing compounds.txt file are the same as
                # those contained in the .inp file
                """
                equal_compounds_number = len(
                    [
                        c1
                        for c1, c2 in zip(compounds_name, self._compounds_name)
                        if c1 == c2
                    ]
                )
                if equal_compounds_number == self._compounds_number:
                    return 
                """
                if collections.Counter(compounds_name) == collections.Counter(
                    self._compounds_name
                ):
                    return

        data = []
        compounds_name = list(set(obs.name for obs in self._observables_ref))
        compounds_name.sort()
        for name in compounds_name:
            observables = []
            for o in self._observables_ref:
                if o.name == name:
                    observable_name = (
                        f"{o.observable}_{o.indexes}"
                        if o.indexes != "0"
                        else f"{o.observable}"
                    )
                    if not self.obj_func_type:
                        observables.append(get_weight_squared_obj_func(observable_name))
                    else:
                        observables.append(
                            get_weight_relative_obj_func(observable_name)
                        )

            observables = list(set(observables))
            observables.sort()

            data.append(f"{name}  {''.join(observables)}\n")

        with open(self._compounds_file_name, "w") as compounds_file:
            compounds_file.writelines(data)

    def _fill_compound_weight_list(self) -> None:
        """
        Read the weight values from the compounds.txt file
        """
        try:
            with open(self._compounds_file_name) as compounds_file:
                self.compound_weight_list = list(map(str.strip, compounds_file))
        except OSError:
            self.compound_weight_list = []

    def _get_path_output_file(self, file_name: str) -> str:
        """
        Helper function to get the relative file name considering the provided output directory.

        :param file_name: Desired file name
        :return: Joined file name with the output directory
        """
        return os.path.join(self.output_dir, file_name)

    def _read_inp_gulp(self):
        """

        :return:
        """
        # Get the content of the .inp file
        with open(self.inp_file_name) as file:
            self.buffer_inp_file = file.readlines()

        # Check if the fit keywords is present in th .inp file
        if not self.is_analysis_gout:
            if "fit " in self.buffer_inp_file[0]:
                print(f"Please remove the 'fit' keyword from the .inp file.")
                raise

        # The number of structures is obtained by counting the 'name' word in the .inp file
        # Avoid using the 'name' word for comments in the .inp file
        try:
            self._compounds_name = [
                b.split()[1]
                for b in self.buffer_inp_file
                if "name " in b.lower() and not ("#" in b)
            ]
        except:
            print(
                f"The name of the compounds could not be read properly from the .inp file."
            )
            raise

        self._compounds_number = len(self._compounds_name)
        if self._compounds_number == 0:
            print(
                "The 'name' GULP flag that identifies the compounds could not be found in the .inp file"
            )
            raise

    def _get_params_gulp(self) -> None:
        """
        Extract the GULP parameters chosen by the user through the 'parameter_lowerbound_upperbound' syntax
        contained in the .inp file (inp_file_name). Please consult the README.md file for more details.
        For instance, considering the below potentials extracted from the example1.gin provided by the GULP program
            buckingham
            Al shel O shel  2409.505 0.2649  0.00 0.0 10.0
            O  shel O shel    25.410 0.6937 32.32 0.0 12.0
        If one wants to fit the parameters regarding the O atom in range of [1.0, 50.0], [0.1, 1.0], and
        [10.0, 50.0], respectively, it would only be necessary to define the following in the .inp file:
            buckingham
            Al shel O shel  2409.505          0.2649          0.00           0.0 10.0
            O  shel O shel    25.410_1.0_50.0 0.6937_0.1_1.0 32.32_10.0_50.0 0.0 12.0
        """
        if not self.buffer_inp_file:
            raise

        self.params = []
        self.lower_params = []
        self.upper_params = []

        # Index used to prepare the list of the adjustable parameters
        idx = 0

        # Find the 'parameter_lowerbound_upperbound' syntax in .inp file in order to detect all the adjustable
        # parameters and their corresponding values for the lower and upper bounds.
        for i, buffer in enumerate(self.buffer_inp_file):
            # Continue if the line does not have 'parameter_lowerbound_upperbound'
            if len(iter_find(buffer, "_")) < 2:
                continue

            line = []
            for string in buffer.split():
                if len(iter_find(string, "_")) == 2:
                    # Extract the adjustable parameter and the corresponding lower and upper bounds
                    # if the 'parameter_lowerbound_upperbound' syntax is found
                    try:
                        param, lower, upper = tuple(map(float, string.split("_")))
                    except:
                        print(
                            f"It was not possible to obtain the parameter and the bounds from '{string}'."
                        )
                        raise

                    self.params.append(param)
                    self.lower_params.append(lower)
                    self.upper_params.append(upper)

                    # Replace 'parameter_lowerbound_upperbound' by PARAM_idx.
                    # PARAM_idx is consistent with the params variable
                    line.append(f"  PARAM_{idx}  ")
                    idx += 1
                else:
                    line.append(f" {string}")

            # Add all PARAM_idx found in the same line to the list that stores the content of the .inp file
            self.buffer_inp_file[i] = f"{''.join(line)}\n"

        if len(self.params) == 0:
            print(
                "Apparently no parameter to be adjusted has been defined. Please use the following "
                "syntax to define the adjustable parameters: 'parameter_lowerbound_upperbound'"
            )
            raise

    def _get_observables(self) -> None:
        """
        Get the observables from the .inp file to be considered in the process of parameterization for the
        calculations of the objective function
        """

        if not self.buffer_inp_file:
            return

        iter_buffer = iter(self.buffer_inp_file)
        for buffer in iter_buffer:
            if not buffer.strip() or "#" in buffer:
                continue

            # Get the name of the compound to specify each observable found in the .inp file
            if "name" in buffer.lower():
                name = buffer.split()[1]

            # The cell observable needs a specific treatment and will be read in the
            # "Comparison of initial and final structures" section contained in the GULP output file

            if "observable" in buffer.lower():
                for b in iter_buffer:
                    # Ignore the line if it contains '#'
                    if not b.strip() or "#" in b:
                        continue

                    if "end" in b:
                        break

                    s = b.split()
                    # if b is not a digit this means that it is the noun of the observable
                    if not s[-1].replace(".", "").replace("-", "").isdigit():
                        observable = b.strip()
                        continue

                    factor = 1.0
                    if "energy" in observable:
                        factor = AU2EV if "au" in observable else 1.0
                        observable = f"{observable.split()[0]}"

                    # Reference value to be compared to the one calculated using the set of parameters
                    value = (
                        float(s[0]) * factor
                        if "energy" in observable
                        else float(s[-1]) * factor
                    )

                    # Get the indexes which identify the corresponding value in the matrix
                    indexes = " ".join(s[:-1]) if len(s) > 2 else "0"

                    self._observables_ref.append(
                        ObservablesData(name, observable, indexes, value)
                    )

    def _get_matrix_data(self, gout_file: TextIO, observable: str) -> list:
        """
        Get the desired data from the output file created by GULP

        :param gout_file: Output file created by GULP from the set of parameters provided
        by the SciPy minimization algorithm
        :param observable: Desired observable to be stored in the 'self._observables_calc' variable
        :return: List containing the data related to the desired property.
        """
        next(gout_file)
        next(gout_file)

        matrix = []
        try:
            # Read the title of the column in order to get the matrix dimension
            s = next(gout_file).split()
            if s[0] == "Indices":
                s = s[1:]  # remove the 'Indices' from the count

            next(gout_file)

            for i in range(len(s)):
                # Ignore the first column, where the title of the row is stored
                line = next(gout_file).split()[1:]
                # Store the value and its corresponding row and column index.
                # This index is important to compare the calculated value with the respective reference value.
                for j, l in enumerate(line):
                    matrix.append((f"{i + 1} {j + 1}", float(l)))
        except:
            print(f"It not possible to read the {observable} data.\n")
            return []

        return matrix

    def _insert_observable(self, gout_file: TextIO, name: str, observable: str) -> bool:
        """
        Store the observable calculated by GULP considering the set of parameters provided
        by the SciPy minimization method

        :param gout_file: Output file created by GULP from the set of parameters provided
        by the SciPy minimization algorithm
        :param name: Name of the compound to be inserted together the observable
        :param observable: Desired observable to be stored in the 'self._observables_calc' variable
        :return: True if the observable is read successfully and False otherwise
        """
        values = self._get_matrix_data(gout_file, observable)
        # If the 'values' variable is empty this means that the corresponding data was not read properly.
        if not values:
            self._set_max_objective_function()
            self.get_detected_error += f"{name} GULP terminated with an error\n"
            return False

        for v in values:
            self._observables_calc.append(ObservablesData(name, observable, v[0], v[1]))

        return True

    def _insert_cell_observable(self, gout_file: TextIO, name: str) -> bool:
        """
        Get the calculated lattice parameters from the output file created by GULP

        :param gout_file: Output file created by GULP from the set of parameters provided
        by the SciPy minimization algorithm
        :param name: Name of the compound to be inserted together the observable
        :return: True if the observable is read successfully and False otherwise
        """
        try:
            [next(gout_file) for _ in range(4)]

            for line in gout_file:
                if "----------------" in line:
                    break

                str_list = line.split()
                index = "_".join(str_list[:-5])
                value = float(str_list[-4])

                self._observables_calc.append(
                    ObservablesData(name, "cell", index, value)
                )
            return True
        except:
            self._set_max_objective_function()
            self.get_detected_error += "GULP terminated with an error\n"

            return False

    def _export_next_gulp_input(self, params: list) -> None:
        """
        Export a GULP .gin file containing the new set of adjustable parameters of the interatomic potentials

        :param params: List of parameters provided by the minimization algorithm
        """
        # Get the content of the .inp file and replace all PARAM_# text by the corresponding parameter provided by the
        # minimization method since PARAM_# is consistent with the params variable
        buffer_aux = list(self.buffer_inp_file)
        for i, j in itertools.product(range(len(buffer_aux)), range(len(params))):
            buffer_aux[i] = buffer_aux[i].replace(f"PARAM_{j} ", f"{params[j]:.4f}")

        # As soon as the buffer_aux variable is prepared, the .gin file is created and will be executed by GULP
        # in the objective function calculation
        with open(self.gin_file_name, "w") as gin_file:
            gin_file.writelines(buffer_aux)

    def _append_points_file(self, params: list) -> None:
        """
        Add the set of parameters to POINTS_FILE

        :param params: List of parameters provided by the SciPy minimization algorithm
        """
        with open(self.points_file, "a") as file:
            file.write(f"{self.point_number}\n")
            for i, param in enumerate(params):
                file.write(f"PARAM_{i:<10}{param:>16.4f}\n")

            file.write(f"Objective Function = {self._objective_function:.16e}\n")

    def _append_analysis_file(self):
        """
        Add the main information extracted from the .gout file in ANALYSIS_FILE. Such an information
        is used in the objective function calculation
        """
        title = "         ref        calc    error^2    %P.R.E."
        with open(self.analysis_file, "a") as file:
            file.write(
                f"\n{'*' * 61}\n"
                f"{'POINT NUMBER':>35} = {self.point_number}"
                f"\n{'*' * 61}"
            )

            last_name = self._data_list[0].name
            file.write(f"\n{last_name:<15}{title}\n")
            relative_error_data = []
            for data in self._data_list:
                if last_name != data.name:
                    file.write(f"{'-' * 61}\n{data.name:<15}{title}\n")
                    last_name = data.name

                indexes = f"({data.indexes})"
                relative_error_data.append(data.relative_error)
                file.write(
                    f"{data.observable[:4]}{indexes:<15}"
                    f"{data.ref:>10.4f} {data.calc:>10.4f} {data.squared_error:>11.4e} {data.relative_error:>7.2f}\n"
                )

            file.write(f"Objective Function = {self._objective_function:.6e}")

            # 'Average Percentage Relative Error' might differ from the 'Objective Function' because the
            # latter might take into account weights provided by the user
            average = sum(error for error in relative_error_data) / len(
                relative_error_data
            )
            file.write(f"\nAverage Percentage Relative Error = {average:.6e}\n")

            # If the calculated gradient is larger than SKIP_GRAD or any imaginary frequency is detected.
            # As the ANALYSIS_FILE stores only the best results, this condition occurs only if
            # the first parameterization cycle is subject to a special condition
            if self.get_detected_error.strip():
                file.write(f"{self.get_detected_error}\n")

    def _exec_gulp_command(self, timeout: int = GULP_TIME_OUT) -> None:
        """
        Run the GULP calculation using the .gin file. The GULP program specified by the gulp_path parameter is used.
        The default GULP location is 'c:\gulp\gulp.exe' on Windows and '~/gulp/Src/gulp' on Linux

        :param timeout: Chosen timeout to stop a GULP calculation, the default value is 15 seconds
        """

        path = os.path.dirname(self.gin_file_name)

        base_name = os.path.basename(self.gin_file_name.replace(".gin", ""))
        args = (self.gulp_path, base_name, base_name)

        return exec_command(args, path, timeout)

    def _set_max_objective_function(self) -> None:
        """
        Define the default maximum objective function of 1e7
        """
        self._objective_function = MAX_OBJECTIVE_VALUE

    def _setup_if_new_min_objective_function(self) -> None:
        """
        Add the information extracted from .gout file to ANALYSIS_FILE if the calculated objective function
        is a new local minimum
        """

        # If an exception (gradient > --skip_grad or presence of imaginary frequency) occurs
        # in the first parameterization cycle, the parameters are not ignored by ParamGULP.
        # In this case, the objective function is evaluated and the parameterization continues
        # in order to try finding a set of parameters that does not violate such conditions.
        # Case this occurs, this new objective function will be considered as the reference
        special_condition = (
            not self._special_condition_disabled
            and not self.get_detected_error.strip()
            and self._objective_function < MAX_OBJECTIVE_VALUE
        )

        if (
            self._objective_function <= self._objective_function_min
            or special_condition
        ):
            # Case a set of parameters that does not violate the mentioned conditions is found,
            # this new objective function will be considered as the reference
            if special_condition:
                self._special_condition_disabled = True

            self._objective_function_min = self._objective_function
            self._point_number_min = self._point_number
            self._append_analysis_file()

    def _read_gout_file(self) -> bool:
        """
        Read the data from the output file created by GULP.
        """
        self._observables_calc = []
        self.get_detected_error = ""

        structure_number = 0
        with open(self.gout_file_name) as gout_file:
            for line in gout_file:
                if "error opening file" in line:
                    print(line)
                    self._calc_objective_function()

                if "ERROR :" in line:
                    return False

                if "Output for configuration" in line:
                    name = line.split()[6]
                    structure_number += 1

                if "Final Gnorm" in line:
                    gnorm = split2float(line, 3, 100)

                    # Ignore the evaluated set of parameters if the gradient is bigger than --skip_grad
                    if gnorm > self.skip_grad:
                        self.get_detected_error += (
                            f"Gradient norm of {name} is more than {self._skip_grad}"
                        )
                        if self.point_number > 1:
                            return False

                if "Comparison of initial and final structures" in line:
                    if not self._insert_cell_observable(gout_file, name):
                        return False

                if "Elastic Constant Matrix:" in line:
                    if not self._insert_observable(gout_file, name, "elastic"):
                        return False

                if "Static dielectric constant tensor" in line:
                    if not self._insert_observable(gout_file, name, "sdlc"):
                        return False

                if "High frequency dielectric constant tensor" in line:
                    if not self._insert_observable(gout_file, name, "hfdlc"):
                        return False

                if "Shear Modulus" in line:
                    try:
                        self._observables_calc.append(
                            ObservablesData(
                                name, "shear_modulus", "0", split2float(line, 4)
                            )
                        )
                    except:
                        return False

                if "Total lattice energy" in line and "eV" in line:
                    try:
                        self._observables_calc.append(
                            ObservablesData(name, "energy", "0", split2float(line, 4))
                        )
                    except:
                        self._set_max_objective_function()
                        return False

                if "Frequencies" in line:
                    next(gout_file)
                    try:
                        frequency = split2float(next(gout_file), 0, -1.0)
                        # Ignore the evaluated set of parameters if any imaginary frequency is detected
                        if frequency < 0.0:
                            self.get_detected_error += (
                                f"The lowest frequency value of {name} is imaginary.\n"
                            )
                            if self.point_number > 1:
                                self._set_max_objective_function()
                                return False
                    except:
                        self._set_max_objective_function()
                        return False

                if "Job Finished" in line:
                    break
            else:
                # Return False if the GULP calculation is not finished properly
                self.get_detected_error += "GULP calculation was not finished properly."
                return False

        if structure_number < self._compounds_number:
            return False

        return True

    def get_weight(self, name, observable, indexes):
        """
        Return the weight value which is considered in the calculations of the objective function calculation

        :param name: Name of the compound whose weight value must be returned
        :param observable: Observable whose weight value must be returned
        :param indexes: Contain additional information for the observable
        :return: Weight value of the compound for the corresponding observable
        """
        weight = 1.0
        for data in self.compound_weight_list:
            if name in data:
                if "cell" in observable:
                    if "_x" in indexes or "_y" in indexes or "_z" in indexes:
                        observable = "cell_xyz"
                    else:
                        observable = "cell_abc"

                weight = float(data.split(observable)[1].split()[0])
                break

        return weight

    def _calc_objective_function(self, is_exec_gulp: bool = True):
        """
        Calculate the objective function by comparing all observables found in the .inp file and the
        corresponding values read from the output file created by GULP
        """
        self._last_objective_function = self._objective_function

        # Execute GULP using the .gin file as input
        if is_exec_gulp:
            self._exec_gulp_command(self.gulp_timeout)

        if not self._read_gout_file():
            self._set_max_objective_function()
            print(self.get_detected_error)
            return

        self._data_list = []
        # Compare the reference value with the respective calculated value
        for o_ref in self._observables_ref:
            for o_calc in self._observables_calc:
                o_calc_data = (o_calc.name, o_calc.observable, o_calc.indexes)
                if (o_ref.name, o_ref.observable, o_ref.indexes) == o_calc_data:
                    # Percentage relative error
                    relative_error = (
                        (abs(o_ref.value - o_calc.value) / abs(o_ref.value)) * 100
                        if o_ref.value != 0
                        else 0.0
                    )

                    # Squared differences
                    squared_error = (o_ref.value - o_calc.value) ** 2

                    self._data_list.append(
                        ObjectiveFunctionData(
                            name=o_ref.name,
                            observable=o_ref.observable,
                            indexes=o_ref.indexes,
                            ref=o_ref.value,
                            calc=o_calc.value,
                            squared_error=squared_error,
                            relative_error=relative_error,
                        )
                    )

        # Calculate the objective function (sum of percentage relative errors or sum of squared differences)
        # for all considered compounds, taking into account the weight value for each compound
        self._objective_function = sum(
            (d.squared_error if self.obj_func_type == 0 else d.relative_error)
            * self.get_weight(d.name, d.observable, d.indexes)
            for d in self._data_list
        )
