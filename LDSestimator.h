#pragma once

#include "TreeLDS.h"

double estimateLDS(  //initial values
	TreeLDS& t,
	Eigen::MatrixXd& A,
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar,
	std::ofstream& outFile
);
double estimateLDS(  //initial values
	ForestLDS& t,
	Eigen::MatrixXd& A,
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar,
	std::ofstream& outFile
);

double carpetBombing(  //parameter seeking, return the final log-likelihood
	TreeLDS& t,
	int seekNo, //how many times execute parameter seeking
	double parameterMin, //parameter boundaries
	double parameterMax,
	double covarMin,
	double covarMax,
	Eigen::MatrixXd& A, //store the last result
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar,
	bool normalization = false
);

double carpetBombing(  //parameter seeking, return the final log-likelihood
	ForestLDS& t,
	int seekNo, //how many times execute parameter seeking
	double parameterMin, //parameter boundaries
	double parameterMax,
	double covarMin,
	double covarMax,
	Eigen::MatrixXd& A, //store the last result
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar,
	bool normalization = false
);

double carpetBombing(  //parameter seeking, return the final log-likelihood
	TreeLDS& t,
	int seekNo, //how many times execute parameter seeking
	double parameterMin, //parameter boundaries
	double parameterMax,
	double covarMin,
	double covarMax,
	Eigen::MatrixXd& A, //store the last result
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::VectorXd& meanV,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar,
	bool normalization = false
);

double carpetBombing(  //parameter seeking, return the final log-likelihood
	ForestLDS& t,
	int seekNo, //how many times execute parameter seeking
	double parameterMin, //parameter boundaries
	double parameterMax,
	double covarMin,
	double covarMax,
	Eigen::MatrixXd& A, //store the last result
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::VectorXd& meanV,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar,
	bool normalization = false
);

void matOut(Eigen::MatrixXd A, std::ofstream& out); //messy, need to be refactored sometime...
void vecOut(Eigen::VectorXd A, std::ofstream& out);

void symmetrize(Eigen::MatrixXd& m);