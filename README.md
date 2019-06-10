# Lineage EM algorithm (LEM)

## Overview
A tool for the latent-variable estimation from cell lineage trees. Implimentation of [https://doi.org/10.1101/488981].


## Demo
```cpp
//estimate continous latent-variable from lineage tree
void EstimatorLDS(std::string experiment, std::string resultFile, int n) { 
	//hyperparameters for the estimation
	int seekNo = 1000; //# of initial values
	double parameterMin = -5.0; //for initial values of parameters
	double parameterMax = 5.0;
	double covarMin = 0.1;
	double covarMax = 5.0;


	//initial condition
	int m = 1;
	Mat initVar(n, n);
	Mat A(n, n);
	Mat C(m, n);
	Mat sigmaW(n, n);
	Mat sigmaV(m, m);
	Vec initState(n);


	//Read
	ForestLDS forest(n, m);
	forest.load("FILENAME");

	//Lineage EM algorithm with multiple initial values (carpet bombing search).
	double logLik = carpetBombing(forest, seekNo, parameterMin, parameterMax, covarMin, covarMax, A, C, sigmaW, sigmaV, initState, initVar, true);

	//output parameters
	std::ofstream outFile(resultFile);

	outFile << logLik << " ";
	outFile << "A: ";
	matOut(A, outFile);
	outFile << "C: ";
	matOut(C, outFile);
	outFile << "sigmaW: ";
	matOut(sigmaW, outFile);
	outFile << "sigmaV: ";
	matOut(sigmaV, outFile);
	outFile << "rootMean: ";
	vecOut(initState, outFile);
	outFile << "rootVar: ";
	matOut(initVar, outFile);
	outFile << std::endl;

	//output latent variables
	forest.smoothing(A, C, sigmaW, sigmaV, initState, initVar);
	forest.output();
}
```

See also test.cpp, execute-estimator.cpp for other examples.


## Requirement
C++14, Boost C++ library 1.64.0, and Eigen 3.3.3.

## Usage
The survivorship bias is NOT corrected in this program.
You should correct it manually or by combining this program with other programs.

## Format of input files for trees
discrete latent-variable
```
#first line
Number_of_nodes Number_of_states
#iteration of one of the following two lines.
id_of_parent 0 division_time #parents must appear in the upper line. States are represented by 0-origin integers. ID of the cell = # of line (0-origin).
id_of_parent 1 state division_time #for leaf nodes whose state is known 
#let state = -1 for unkonwn states
```

continuous latent-variable
```
#iteration of the following row entry:
id_of_parent  is_Leaf? observed_variable (m components) #is_leaf is represented by boolean (0 or 1). observed_variable is typically division time (m = 1)
```

## Source codes
The following two codes are the core of LEM.
- Tree.cpp: Class for discrete latent-variable tree (includeing E-step of LEM).
- TreeLDS.cpp: Class for continous latent-variable tree (including E-step and M-step of LEM).

Other codes.
- estimator.cpp: Implementation of the LEM for discrete latent-variable for spefic models.
- execute-estimator.cpp: examples for LEM for discrete latent-variables.
- estimatorLDS.cpp: LEM for continous latent-variables.
- test.cpp: other examples.
- dataGenerator.cpp: generate synthetic lineage trees.
