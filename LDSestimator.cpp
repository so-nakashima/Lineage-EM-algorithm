#include "TreeLDS.h"
#include "assert.h"
#include <fstream>
#include <iostream>
#include <queue>
#include <Eigen/Core>
#include <Eigen/LU>
#include <limits>
#include <random>

using namespace Eigen;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;

const int MAXITER = 300;
const double epsilon = 1.0e-5;

void symmetrize(Mat& m) {
	assert(m.rows() == m.cols());
	int n = m.rows();

	for (int i = 0; i != n; i++) {
		for (int j = i + 1; j < n; j++) {
			m(i, j) = 0.5 * (m(i, j) + m(j, i));
			m(j, i) = m(i, j);
		}
	}
}

void matOut(Mat A, std::ofstream& out) {
	for (int i = 0; i != A.rows(); i++) {
		for (int j = 0; j != A.cols(); j++) {
			out << A(i, j) << ", ";
		}
		
	}
}

void vecOut(Vec A, std::ofstream& out) {
	for (int i = 0; i != A.size(); i++) {
		out << A[i] << ", ";
	}
}


double estimateLDS(  //return log-likelihood
	TreeLDS& t,
	Eigen::MatrixXd& A, //initial values
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar,
	std::ofstream& outFile
) {
	double prevLogLik = 0;

	Mat prevA = A;
	Mat prevC = C;
	Mat prevSigmaW = sigmaW;
	Mat prevsigmaV = sigmaV;
	Mat prevRootMean = rootMean;
	Mat prevRootVar = rootVar;

	for (int i = 0; i != MAXITER; i++) {
		//E-step

		t.smoothing(A, C, sigmaW, sigmaV, rootMean, rootVar);

		//Check stop condition
		double logLik = t.logLik();
		double LogLikeRatio = (logLik - prevLogLik) / abs(prevLogLik + epsilon); //to avoid 0-division
		//double LogLikDiff = logLik - prevLogLik;
		//if (abs(LogLikDiff) < epsilon)
		if (i != 0 && (isnan(logLik) || LogLikeRatio < epsilon))
			break;
		else
			prevLogLik = logLik;

		//output
		outFile << t.logLik() << " ";
		outFile << "A: ";
		matOut(A, outFile);
		outFile << "C: ";
		matOut(C, outFile);
		outFile << "sigmaW: ";
		matOut(sigmaW, outFile);
		outFile << "sigmaV: ";
		matOut(sigmaV, outFile);
		outFile << "rootMean: ";
		vecOut(rootMean, outFile);
		outFile << "rootVar: ";
		matOut(rootVar, outFile);
		outFile << std::endl;

		//store the last parameters
		prevA = A;
		prevC = C;
		prevSigmaW = sigmaW;
		prevsigmaV = sigmaV;
		prevRootMean = rootMean;
		prevRootVar = rootVar;

		//Update (M-step)
		t.update(A, C, sigmaW, sigmaV, rootMean, rootVar);
	}

	//return the last value before stopped
	A = prevA;
	C = prevC;
	sigmaW = prevSigmaW;
	sigmaV = prevsigmaV;
	rootMean = prevRootMean;
	rootVar = prevRootVar;

	//return log-likelihood
	return prevLogLik;
}

double estimateLDS(  //return log-likelihood
	ForestLDS& t,
	Eigen::MatrixXd& A, //initial values
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar,
	std::ofstream& outFile
) {
	double prevLogLik = 0;

	Mat prevA = A;
	Mat prevC = C;
	Mat prevSigmaW = sigmaW;
	Mat prevsigmaV = sigmaV;
	Mat prevRootMean = rootMean;
	Mat prevRootVar = rootVar;

	for (int i = 0; i != MAXITER; i++) {
		//E-step

		t.smoothing(A, C, sigmaW, sigmaV, rootMean, rootVar);

		//Check stop condition
		double logLik = t.logLik();
		double LogLikeRatio = (logLik - prevLogLik) / abs(prevLogLik + epsilon); //to avoid 0-division
																				 //double LogLikDiff = logLik - prevLogLik;
																				 //if (abs(LogLikDiff) < epsilon)
		if (i != 0 && (isnan(logLik) || LogLikeRatio < epsilon))
			break;
		else
			prevLogLik = logLik;

		//output
		outFile << t.logLik() << " ";
		outFile << "A: ";
		matOut(A, outFile);
		outFile << "C: ";
		matOut(C, outFile);
		outFile << "sigmaW: ";
		matOut(sigmaW, outFile);
		outFile << "sigmaV: ";
		matOut(sigmaV, outFile);
		outFile << "rootMean: ";
		vecOut(rootMean, outFile);
		outFile << "rootVar: ";
		matOut(rootVar, outFile);
		outFile << std::endl;

		//store the last parameters
		prevA = A;
		prevC = C;
		prevSigmaW = sigmaW;
		prevsigmaV = sigmaV;
		prevRootMean = rootMean;
		prevRootVar = rootVar;

		//Update (M-step)
		t.update(A, C, sigmaW, sigmaV, rootMean, rootVar);
	}

	//return the last value before stopped
	A = prevA;
	C = prevC;
	sigmaW = prevSigmaW;
	sigmaV = prevsigmaV;
	rootMean = prevRootMean;
	rootVar = prevRootVar;

	//return log-likelihood
	return prevLogLik;
}

double estimateLDS(  //return log-likelihood
	TreeLDS& t,
	Eigen::MatrixXd& A, //initial values
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar
) {
	double prevLogLik;

	Mat prevA = A;
	Mat prevC = C;
	Mat prevSigmaW = sigmaW;
	Mat prevsigmaV = sigmaV;
	Mat prevRootMean = rootMean;
	Mat prevRootVar = rootVar;

	for (int i = 0; i != MAXITER; i++) {
		//E-step

		t.smoothing(A, C, sigmaW, sigmaV, rootMean, rootVar);

		//Check stop condition
		double logLik = t.logLik();
		double LogLikeRatio;
		if (i != 0)
			LogLikeRatio = (logLik - prevLogLik) / abs(prevLogLik + epsilon); //to avoid 0-division
																				 
		if (i != 0 && (isnan(logLik) || LogLikeRatio < epsilon))
			break;
		else
			prevLogLik = logLik;

		//store the last parameters
		prevA = A;
		prevC = C;
		prevSigmaW = sigmaW;
		prevsigmaV = sigmaV;
		prevRootMean = rootMean;
		prevRootVar = rootVar;

		//Update (M-step)
		t.update(A, C, sigmaW, sigmaV, rootMean, rootVar);
	}

	//return the last value before stopped
	A = prevA;
	C = prevC;
	sigmaW = prevSigmaW;
	sigmaV = prevsigmaV;
	rootMean = prevRootMean;
	rootVar = prevRootVar;

	//return log-likelihood
	return prevLogLik;
}

double estimateLDS(  //return log-likelihood
	ForestLDS& t,
	Eigen::MatrixXd& A, //initial values
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar
) {
	double prevLogLik;

	Mat prevA = A;
	Mat prevC = C;
	Mat prevSigmaW = sigmaW;
	Mat prevsigmaV = sigmaV;
	Mat prevRootMean = rootMean;
	Mat prevRootVar = rootVar;

	for (int i = 0; i != MAXITER; i++) {
		//E-step

		t.smoothing(A, C, sigmaW, sigmaV, rootMean, rootVar);

		//Check stop condition
		double logLik = t.logLik();
		double LogLikeRatio;
		if (i != 0)
			LogLikeRatio = (logLik - prevLogLik) / abs(prevLogLik + epsilon); //to avoid 0-division

		if (i != 0 && (isnan(logLik) || LogLikeRatio < epsilon))
			break;
		else
			prevLogLik = logLik;

		//store the last parameters
		prevA = A;
		prevC = C;
		prevSigmaW = sigmaW;
		prevsigmaV = sigmaV;
		prevRootMean = rootMean;
		prevRootVar = rootVar;

		//Update (M-step)
		t.update(A, C, sigmaW, sigmaV, rootMean, rootVar);
	}

	//return the last value before stopped
	A = prevA;
	C = prevC;
	sigmaW = prevSigmaW;
	sigmaV = prevsigmaV;
	rootMean = prevRootMean;
	rootVar = prevRootVar;

	//return log-likelihood
	return prevLogLik;
}

double estimateLDS(  //return log-likelihood
	TreeLDS& t,
	Eigen::MatrixXd& A, //initial values
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::VectorXd& meanV,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar
) {
	double prevLogLik;

	Mat prevA = A;
	Mat prevC = C;
	Mat prevSigmaW = sigmaW;
	Vec prevMeanV = meanV;
	Mat prevsigmaV = sigmaV;
	Mat prevRootMean = rootMean;
	Mat prevRootVar = rootVar;

	for (int i = 0; i != MAXITER; i++) {
		//E-step

		t.smoothing(A, C, sigmaW, meanV, sigmaV, rootMean, rootVar);

		//Check stop condition
		double logLik = t.logLik();
		double LogLikeRatio;
		if (i != 0)
			LogLikeRatio = (logLik - prevLogLik) / abs(prevLogLik + epsilon); //to avoid 0-division

		if (i != 0 && (isnan(logLik) || LogLikeRatio < epsilon))
			break;
		else
			prevLogLik = logLik;

		//store the last parameters
		prevA = A;
		prevC = C;
		prevSigmaW = sigmaW;
		prevMeanV = meanV;
		prevsigmaV = sigmaV;
		prevRootMean = rootMean;
		prevRootVar = rootVar;

		//Update (M-step)
		t.update(A, C, sigmaW, meanV, sigmaV, rootMean, rootVar);
	}

	//return the last value before stopped
	A = prevA;
	C = prevC;
	sigmaW = prevSigmaW;
	meanV = prevMeanV;
	sigmaV = prevsigmaV;
	rootMean = prevRootMean;
	rootVar = prevRootVar;

	//return log-likelihood
	return prevLogLik;
}

double estimateLDS(  //return log-likelihood
	ForestLDS& t,
	Eigen::MatrixXd& A, //initial values
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::VectorXd& meanV,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar
) {
	double prevLogLik;

	Mat prevA = A;
	Mat prevC = C;
	Mat prevSigmaW = sigmaW;
	Vec prevMeanV = meanV;
	Mat prevsigmaV = sigmaV;
	Mat prevRootMean = rootMean;
	Mat prevRootVar = rootVar;

	for (int i = 0; i != MAXITER; i++) {
		//E-step

		t.smoothing(A, C, sigmaW, meanV, sigmaV, rootMean, rootVar);

		//Check stop condition
		double logLik = t.logLik();
		double LogLikeRatio;
		if (i != 0)
			LogLikeRatio = (logLik - prevLogLik) / abs(prevLogLik + epsilon); //to avoid 0-division

		if (i != 0 && (isnan(logLik) || LogLikeRatio < epsilon))
			break;
		else
			prevLogLik = logLik;

		//store the last parameters
		prevA = A;
		prevC = C;
		prevSigmaW = sigmaW;
		prevMeanV = meanV;
		prevsigmaV = sigmaV;
		prevRootMean = rootMean;
		prevRootVar = rootVar;

		//Update (M-step)
		t.update(A, C, sigmaW, meanV, sigmaV, rootMean, rootVar);
	}

	//return the last value before stopped
	A = prevA;
	C = prevC;
	sigmaW = prevSigmaW;
	meanV = prevMeanV;
	sigmaV = prevsigmaV;
	rootMean = prevRootMean;
	rootVar = prevRootVar;

	//return log-likelihood
	return prevLogLik;
}

void matGeneration(Mat& A, double parameterMin, double parameterMax) {
	//random
	static std::mt19937 gen{ std::random_device{}() };
	static std::uniform_real_distribution<double> dist(parameterMin, parameterMax);
	
	//generation
	for (int i = 0; i != A.rows(); i++)
		for (int j = 0; j != A.cols(); j++)
			A(i, j) = dist(gen);
}

void normalizedMatGeneration(Mat& A, double parameterMin, double parameterMax) {
	//random
	static std::mt19937 gen{ std::random_device{}() };
	static std::uniform_real_distribution<double> dist(parameterMin, parameterMax);

	//generation
	while (true) {
		for (int i = 0; i != A.rows(); i++)
			for (int j = 0; j != A.cols(); j++)
				A(i, j) = dist(gen);

		if (abs(A.determinant()) > 1.0e-5)
			break;
	}

	//normalization
	double det = abs(A.determinant());
	A /= pow(det, 1.0 / A.rows());
}

void vecGeneration(Vec& v, double parameterMin, double parameterMax) {
	//random
	static std::mt19937 gen{ std::random_device{}() };
	static std::uniform_real_distribution<double> dist(parameterMin, parameterMax);

	//generation
	for (int i = 0; i != v.size(); i++)
		v[i] = dist(gen);
}

void covarGeneration(Mat& covar, double covarMin, double covarMax, double parameterMin, double parameterMax) {
	//initialization and random
	int n = covar.rows();
	covar = MatrixXd::Zero(n, n);
	static std::mt19937 gen{ std::random_device{}() };
	static std::uniform_real_distribution<double> dist(covarMin, covarMax);
	
	//generation
	for (int i = 0; i != n; i++) {
		Vec mu = VectorXd::Zero(n);
		mu.normalize();
		vecGeneration(mu, parameterMin, parameterMax);
		double coef = dist(gen);
		Mat hoge = mu * mu.transpose();
		covar += coef * hoge;
	}

	//regularize
	double det = covar.determinant();
	if (det < pow(covarMin, n))
		covar *= covarMin / pow(det, 1.0 / n);
}

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
) {
	//assertions
	assert(parameterMin < parameterMax);
	assert(covarMin > 0);
	assert(covarMax > covarMin);
	assert(seekNo >= 0);

	int n = t.n;
	int m = t.m;

	//initialization
	A = MatrixXd::Zero(n, n);
	C = MatrixXd::Zero(m, n);
	sigmaW = MatrixXd::Zero(n, n);
	sigmaV = MatrixXd::Zero(m, m);
	rootVar = MatrixXd::Zero(n, n);
	rootMean = VectorXd::Zero(n);


	//store best values
	Mat bestA;
	Mat bestC;
	Mat bestSigmaW;
	Mat bestsigmaV;
	Mat bestRootMean;
	Mat bestRootVar;
	double bestLogLik = - DBL_MAX;

	for (int i = 0; i != seekNo; i++) {
		//generation
		if (!normalization)
			matGeneration(A, parameterMin, parameterMax);
		else
			normalizedMatGeneration(A, parameterMin, parameterMax);

		matGeneration(C, parameterMin, parameterMax);
		vecGeneration(rootMean, parameterMin, parameterMax);
		covarGeneration(sigmaW, covarMin, covarMax, parameterMin, parameterMax);
		covarGeneration(sigmaV, covarMin, covarMax, parameterMin, parameterMax);
		covarGeneration(rootVar, covarMin, covarMax, parameterMin, parameterMax);

		//estimation
		double logLik = estimateLDS(t, A, C, sigmaW, sigmaV, rootMean, rootVar);


		//update best values
		if (!isnan(logLik) && logLik > bestLogLik) {
			bestA = A;
			bestC = C;
			bestSigmaW = sigmaW;
			bestsigmaV = sigmaV;
			bestRootMean = rootMean;
			bestRootVar = rootVar;
			bestLogLik = logLik;
		}

		//progress report
		if (i % 1 == 0)
			std::cout << i << " th bombing have finished. Current best log likelihood is: " << bestLogLik << std::endl;
	}

	//final output
	A = bestA;
	C = bestC;
	sigmaW = bestSigmaW;
	sigmaV = bestsigmaV;
	rootMean = bestRootMean;
	rootVar = bestRootVar;

	return bestLogLik;
}

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
) {
	//assertions
	assert(parameterMin < parameterMax);
	assert(covarMin > 0);
	assert(covarMax > covarMin);
	assert(seekNo >= 0);

	int n = t.n;
	int m = t.m;

	//initialization
	A = MatrixXd::Zero(n, n);
	C = MatrixXd::Zero(m, n);
	sigmaW = MatrixXd::Zero(n, n);
	sigmaV = MatrixXd::Zero(m, m);
	rootVar = MatrixXd::Zero(n, n);
	rootMean = VectorXd::Zero(n);


	//store best values
	Mat bestA;
	Mat bestC;
	Mat bestSigmaW;
	Mat bestsigmaV;
	Mat bestRootMean;
	Mat bestRootVar;
	double bestLogLik = -DBL_MAX;

	for (int i = 0; i != seekNo; i++) {
		//generation
		if(!normalization)
			matGeneration(A, parameterMin, parameterMax);
		else
			normalizedMatGeneration(A, parameterMin, parameterMax);

		matGeneration(C, parameterMin, parameterMax);
		vecGeneration(rootMean, parameterMin, parameterMax);
		covarGeneration(sigmaW, covarMin, covarMax, parameterMin, parameterMax);
		covarGeneration(sigmaV, covarMin, covarMax, parameterMin, parameterMax);
		covarGeneration(rootVar, covarMin, covarMax, parameterMin, parameterMax);

		//estimation
		double logLik = estimateLDS(t, A, C, sigmaW, sigmaV, rootMean, rootVar);


		//update best values
		if (!isnan(logLik) && logLik > bestLogLik) {
			bestA = A;
			bestC = C;
			bestSigmaW = sigmaW;
			bestsigmaV = sigmaV;
			bestRootMean = rootMean;
			bestRootVar = rootVar;
			bestLogLik = logLik;
		}

		//progress report
		if (i % 1 == 0)
			std::cout << i << " th bombing have finished. Current best log likelihood is: " << bestLogLik << std::endl;
	}

	//final output
	A = bestA;
	C = bestC;
	sigmaW = bestSigmaW;
	sigmaV = bestsigmaV;
	rootMean = bestRootMean;
	rootVar = bestRootVar;

	return bestLogLik;
}

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
) {
	//assertions
	assert(parameterMin < parameterMax);
	assert(covarMin > 0);
	assert(covarMax > covarMin);
	assert(seekNo >= 0);

	int n = t.n;
	int m = t.m;

	//initialization
	A = MatrixXd::Zero(n, n);
	C = MatrixXd::Zero(m, n);
	sigmaW = MatrixXd::Zero(n, n);
	meanV = VectorXd::Zero(m);
	sigmaV = MatrixXd::Zero(m, m);
	rootVar = MatrixXd::Zero(n, n);
	rootMean = VectorXd::Zero(n);


	//store best values
	Mat bestA;
	Mat bestC;
	Mat bestSigmaW;
	Vec bestMeanV;
	Mat bestsigmaV;
	Mat bestRootMean;
	Mat bestRootVar;
	double bestLogLik = -DBL_MAX;

	for (int i = 0; i != seekNo; i++) {
		//generation
		if (!normalization)
			matGeneration(A, parameterMin, parameterMax);
		else
			normalizedMatGeneration(A, parameterMin, parameterMax);

		matGeneration(C, parameterMin, parameterMax);
		vecGeneration(rootMean, parameterMin, parameterMax);
		covarGeneration(sigmaW, covarMin, covarMax, parameterMin, parameterMax);
		vecGeneration(meanV, parameterMin, parameterMax);
		covarGeneration(sigmaV, covarMin, covarMax, parameterMin, parameterMax);
		covarGeneration(rootVar, covarMin, covarMax, parameterMin, parameterMax);

		//estimation
		double logLik = estimateLDS(t, A, C, sigmaW, meanV, sigmaV, rootMean, rootVar);


		//update best values
		if (!isnan(logLik) && logLik > bestLogLik) {
			bestA = A;
			bestC = C;
			bestSigmaW = sigmaW;
			bestMeanV = meanV;
			bestsigmaV = sigmaV;
			bestRootMean = rootMean;
			bestRootVar = rootVar;
			bestLogLik = logLik;
		}

		//progress report
		if (i % 1 == 0)
			std::cout << i << " th bombing have finished. Current best log likelihood is: " << bestLogLik << std::endl;
	}

	//final output
	A = bestA;
	C = bestC;
	sigmaW = bestSigmaW;
	meanV = bestMeanV;
	sigmaV = bestsigmaV;
	rootMean = bestRootMean;
	rootVar = bestRootVar;

	return bestLogLik;
}

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
) {
	//assertions
	assert(parameterMin < parameterMax);
	assert(covarMin > 0);
	assert(covarMax > covarMin);
	assert(seekNo >= 0);

	int n = t.n;
	int m = t.m;

	//initialization
	A = MatrixXd::Zero(n, n);
	C = MatrixXd::Zero(m, n);
	sigmaW = MatrixXd::Zero(n, n);
	meanV = VectorXd::Zero(m);
	sigmaV = MatrixXd::Zero(m, m);
	rootVar = MatrixXd::Zero(n, n);
	rootMean = VectorXd::Zero(n);


	//store best values
	Mat bestA;
	Mat bestC;
	Mat bestSigmaW;
	Vec bestMeanV;
	Mat bestsigmaV;
	Mat bestRootMean;
	Mat bestRootVar;
	double bestLogLik = -DBL_MAX;

	for (int i = 0; i != seekNo; i++) {
		//generation
		if (!normalization)
			matGeneration(A, parameterMin, parameterMax);
		else
			normalizedMatGeneration(A, parameterMin, parameterMax);

		matGeneration(C, 1, 1);
		vecGeneration(rootMean, parameterMin, parameterMax);
		Vec hoge = VectorXd::Zero(n);
		vecGeneration(hoge, 0.1, parameterMax);
		sigmaW = hoge.asDiagonal();
		//covarGeneration(sigmaW, covarMin, covarMax, parameterMin, parameterMax);
		//vecGeneration(meanV, parameterMin, parameterMax);
		meanV[0] = 1.2;
		covarGeneration(sigmaV, covarMin, covarMax, parameterMin, parameterMax);
		covarGeneration(rootVar, covarMin, covarMax, parameterMin, parameterMax);

		//estimation
		double logLik = estimateLDS(t, A, C, sigmaW, meanV, sigmaV, rootMean, rootVar);


		//update best values
		if (!isnan(logLik) && logLik > bestLogLik) {
			bestA = A;
			bestC = C;
			bestSigmaW = sigmaW;
			bestMeanV = meanV;
			bestsigmaV = sigmaV;
			bestRootMean = rootMean;
			bestRootVar = rootVar;
			bestLogLik = logLik;
		}

		//progress report
		if (i % 1 == 0)
			std::cout << i << " th bombing have finished. Current best log likelihood is: " << bestLogLik << std::endl;
	}

	//final output
	A = bestA;
	C = bestC;
	sigmaW = bestSigmaW;
	meanV = bestMeanV;
	sigmaV = bestsigmaV;
	rootMean = bestRootMean;
	rootVar = bestRootVar;

	return bestLogLik;
}