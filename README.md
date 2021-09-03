# ParamGULP Program for Obtaining Interatomic Potential Parameters for GULP
This code is developed by José Diogo L.Dutra et al. I just share the code in the Github.
If you use this code, please cite it: 
```
Dutra, J. D. L., Bispo, T. D., de Freitas, S. M., & Rezende, M. V. D. S. (2021). ParamGULP: An efficient Python code for obtaining interatomic potential parameters for General Utility Lattice Program. Computer Physics Communications, 265, 107996.
```

General Utility Lattice Program (GULP) is an important software commonly used to study several 
properties of solid materials by means of an extensive list of interatomic potentials. 
GULP is available free for academic use and can be downloaded at: 
http://gulp.curtin.edu.au/gulp/request.cfm?rel=download. 
With ParamGULP, we hope to make the parameterization of interatomic potentials more efficient. 

ParamGULP fits the interatomic potential parameters based on some information provided by the user 
in the input file (.inp), as illustrated below for adjusting the parameters of the oxygen atom 
in the range of [1.0, 50.0], [0.1, 1.0], and [10.0, 50.0], respectively:

```
Buckingham 
Al shel O shel 2409.505 0.2649 0.00 0.0 10.0 
O shel O shel 25.410_1.0_50.0 0.6937_0.1_1.0 32.32_10.0_50.0 0.0 12.0
```

ParamGULP is structured in the following steps:

1. From the parameters stored in the .inp file, the process of parameterization is started. 
2. A new GULP input file (.gin) containing the set of parameters to be evaluated is generated.
3. When GULP is performed by using .gin as input file, the correspondent GULP output file (.gout) is created. 
4. The objective function is calculated considering the geometric parameters and the other observables
contained in the .gout file.
5. Using the calculated objective function, the SciPy minimization method implemented into 
ParamGULP generates the new set of parameters.
6. The process of parameterization returns to step 2 and a new parameterization cycle will start.


**Technical Description**

The ParamGULP program is organized in four modules: `common`, `minimize`, `gulp`, and `paramgulp`. 
The corresponding .py source files are stored in the folder named ParamGULP. 

- `common.py`: contains helper functions such as `iter_find` and `split2float` to handle strings, 
and `exec_command` that calls the GULP program. Some important parameters are defined in the `common` module. 
- `minimize.py`: contains the functions responsible to handle the execution of the SciPy minimization methods.
- `gulp.py`: has the class `ParametricGulp` that is designed to handle the GULP input and output files,
 besides running the GULP executable.
- `paramgulp.py`: contains the `main` function and all options supported by ParamGULP. In addition, 
`paramgulp` has some functions that prepare specific files and folders for the calculations. 

For more details regarding each module, the source codes and the paper should be consulted. 


**Installation, Requirements, and Usage**

ParamGULP works on every operating system and does not require installation, 
being necessary to have Python 3 installed along with the SciPy library. Furthermore, 
the GULP program must be installed properly and it is easily compiled using the gfortran compiler. 

For example, ParamGULP can be carried out with the following command:

```
python3 paramgulp.py --method all --input_file ../example/case_study/RFeO3.inp
```

All options supported by ParamGULP are summarized below:

- `--method`: desired minimization method: **all** methods: all; **1**:dual annealing with Nelder-Mead; **2**:dual annealing with Powell. No default value is defined.
- `--input_file`: complete name of the .inp file. No default value is defined.
- `--gulp_path`: location of the GULP executable. By default, *C:/gulp/gulp.exe* and *~/gulp/Src/gulp* 
are used on the Windows® and UNIX operating systems, respectively. 
- `--output_dir`: directory created in the same directory as `--input_file` where the files created by ParamGULP will be saved. 
The default value is *minimize_output*.
- `--start_point`: start point from that the parameterization will be started. By default, the parameters contained in the .inp file are considered as start point (`--start_point 0`).
- `--timeout`: maximum expected time (in seconds) until each optimization method to be automatically aborted. If a value is not provided for --timeout, the optimization method will be aborted when the number of cycles is reached
- `--n_cycles`: maximum number of parameterization cycles. Depending on certain criteria of the minimization method, 
it is possible that this number cannot be reached. The default value is 1000.
- `--n_cores`: number of cores to be used in case of multiple optimizations. All available cores are used with `--n_cores all`. 
The half of the number of the available cores is used by default.
- `--gulp_timeout`: maximum time (in seconds) to wait GULP to finish the calculations. Default value will be equal to 5 times the spent time in the first parameterization cycle.
- `--skip_grad`: if the gradient calculated by GULP is larger than `--skip_grad`, the evaluated parameters will not be considered. 
- `--obj_func`: default value of `--obj_func` is 0 and the objective function based on the sum of squares is used. Otherwise, the objective function is calculated using a sum of percentage relative errors.
- `--tol_params`: tolerance for variations in parameters and default value is 0.001.
- `--tol_cycles`: the parameterization process will be interrupted if after evaluating 200 parameterization cycles, by default, the objective function is not reduced.
- `--tol_ignore`: if different from 0, the tolerance criteria (`--tol_params` and `--tol_cycles`) will not be applied. In this case, the fit procedure is finished if either the number of cycles (`--n_cycles`) or the timeout is reached

The required arguments for ParamGULP are `--method` and `--input_file`, which provide the SciPy minimization algorithm to be used and the complete name of the .inp file. 

**Case Study**

The `examples` folder contains some files used to illustrate the use of ParamGULP. Description of the files included in the subfolders are provided:

`./examples/RFeO3/selected_files`: 
Contains the results obtained for both the initial guess parameters and the parameters optimized with ParamGULP. 

`./examples/RFeO3/selected_files/RFeO3.dat`
File used only to save the initial guess parameters of the interatomic potentials and it is not used by ParamGULP. 

------------------------------------------------------

`./examples/RFeO3/selected_files/initial_guess/RFeO3.gin`
GULP input file containing the initial guess parameters obtained from the literature. 
Please compare ./example/selected_files/RFeO3.dat and RFeO3.gin.

`./examples/RFeO3/selected_files/initial_guess/RFeO3.gout`
GULP output file created from the RFeO3.gin file after running the GULP program. 
RFeO3.gout contains the GULP result for the initial guess parameters.

`./examples/RFeO3/selected_files/initial_guess/RFeO3_initial_guess.inp`
This file shows how to specify the parameters to be adjusted with ParamGULP.

`./examples/RFeO3/selected_files/initial_guess/analysis.txt`
File created by ParamGULP containing the GULP results used to estimate the objective function value.  

------------------------------------------------------

`./examples/RFeO3/selected_files/optimized_parameters/RFeO3.gin`
GULP input file containing the parameters adjusted by ParamGULP.

`./examples/RFeO3/selected_files/optimized_parameters/RFeO3.gout`
GULP output file created from the optimized parameters stored in the RFeO3.gin file.

`./examples/RFeO3/selected_files/optimized_parameters/analysis.txt`
This file stores the GULP result extracted from RFeO3.gout.

`./examples/RFeO3/selected_files/optimized_parameters/points.txt`
File created by ParamGULP to store the parameters provided by the minimization algorithm. 

------------------------------------------------------

`./examples/RFeO3/FIT_PARAM`:
Contains the files used by us to illustrate the use of ParamGULP in the manuscript. 

`./examples/RFeO3/FIT_PARAM/RFeO3.inp`
Input file for ParamGULP containing the specification of the parameters to be adjusted by means of the “*parameter_lowerbound_upperbound*” syntax. 

`./examples/RFeO3/FIT_PARAM/compounds.txt`
The compounds.txt file will be created if it does not exist and contains by default all weight values for all compounds detected in the .inp file. If the result for a given compound is are not being minimized properly, 
the user can try by editing compounds.txt to prioritize such compound in the calculation of the objective function value.  

`./examples/RFeO3/minimize_output`
Such folder stores the results obtained in the exploratory parameterization using the minimization methods available in ParamGULP. The initial guess parameters considered were those obtained from the literature. 

For obtaining these results, the command line below was used:

```
python3 paramgulp.py --method all --input_file ../examples/RFeO3/FIT_PARAM/RFeO3.inp 
--gulp_path /opt/gulp-5.2/gulp-5.2
```

---------------------------------------------------------------------------

`./examples/RFeO3/FIT_GULP`:

Contains the files used by us to illustrate the use of fit procedure contained in the GULP code.

`./examples/RFeO3/FIT_GULP/BFGS`:

Contains the input file for GULP (RFeO3.gin) with the specification of the parameters to be adjusted using the BFGS method of GULP and the corresponding output file (RFeO3.gin). The analysis.txt file created by ParamGULP is contained in the folder and the objective function was calculated with the help of the weight values stored in compounds.txt. 

`./examples/RFeO3/FIT_GULP/SIMPLEX`:

Similar to the folder containing the results obtained by using BFGS of GULP, the SIMPLEX folder contains the corresponding .gin  and .gout files related to the fit with Simplex of GULP. 

---------------------------------------------------------------------------

`./examples/GULP_examples`:
This folder contains some examples provided by the GULP code that were used to compare the performance between ParamGULP and BFGS of GULP. 

`./examples/GULP_examples/FIT_GULP`
Contains the results obtained using both BFGS and Simplex of GULP for each example considered. The `.gin` files correspond to the input file of GULP and .gout are the corresponding output files.  

`./examples/GULP_examples/FIT_PARAM`
Contains the results obtained by using the ParamGULP code and each example considered is contained in its corresponding subfolder.
