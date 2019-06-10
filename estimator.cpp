#include "Tree.h"
#include <string>
#include <vector>
#include "Density.h"
#include <queue>
#include <iostream>
#include <fstream>

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>

//for summation in M-step in tree1Estimator()
double sumSurvival(const Node& n, int type) {
	return n.survivalTime * n.currentDistribution[type];
}
double sumProbWithoutLeaf(const Node& n, int type) {
	if (n.isLeaf())
		return 0.0;
	else
		return n.currentDistribution[type];
}

void tree1EstimatorCore(std::vector<Tree*>& forest, std::vector<std::vector<double>>& transit, double& lambda0, double& lambda1, std::ofstream& resultFile) {
	//Stop condition
	const int maxIteration = 1000;
	const double epsilon = 1.0e-10;
	double logLikeold = 1;


	//report total node, leaf node
	int nodesNo, leavesNo;
	nodesNo = leavesNo = 0;
	for (auto itr = forest.begin(); itr != forest.end(); itr++) {
		nodesNo += (*itr)->size();
		leavesNo += (*itr)->leavesSize();
	}
	std::cout << "No. of nodes is: " << nodesNo << std::endl;
	std::cout << "No. of leaves is: " << leavesNo << std::endl;

	//for iteration
	int count = 0; //count # iterations
	std::vector<Density*> distributions(2, NULL);

	//iteration
	while (count < maxIteration) {
		//initialize distributions
		ExponentialDensity dist0(lambda0), dist1(lambda1);
		distributions[0] = &dist0; distributions[1] = &dist1;

		//E-step
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {
			(*itr)->inference(transit, distributions);
		}

		//report log-likelihood
		double logLik = 0;
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {
			logLik += (*itr)->logLik();
		}
		if(count % 50 == 0)
			std::cout << "The log-Like after " << count << "-th iteration is: " << logLik << std::endl;

		//stop condition on log likelyhood
		if (abs(logLik - logLikeold) < epsilon)
			break;
		else
			logLikeold = logLik;
		resultFile << logLik << " ";

		//M-step
		//for lambdas
		double coef0, coef1, sumtau0, sumtau1; //derivative is of the form coef*(1/lambda) - sumtau = 0. suffix indicates the state
		coef0 = coef1 = sumtau0 = sumtau1 = 0.0;
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {//sum up for all trees
																	 //for sum tau
			std::vector<double> sum1 = (*itr)->sum(sumSurvival);
			sumtau0 += sum1[0]; sumtau1 += sum1[1];
			//for sum coef
			std::vector<double> sum2 = (*itr)->sum(sumProbWithoutLeaf);
			coef0 += sum2[0]; coef1 += sum2[1];
		}

		//for transition Matrix
		std::vector<std::vector<double>> countTransit(2, std::vector<double>(2, 0.0));
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {//sum up for all trees
			std::vector<std::vector<double>> thisTreeCount = (*itr)->transitCoef(transit, distributions);
			countTransit[0][0] += thisTreeCount[0][0];
			countTransit[0][1] += thisTreeCount[0][1];
			countTransit[1][0] += thisTreeCount[1][0];
			countTransit[1][1] += thisTreeCount[1][1];
		}

		//update parameters
		lambda0 = std::max(.005, coef0 / sumtau0); //set minimum value to avoid numerical error
		lambda1 = std::max(.005, coef1 / sumtau1);
		//normalize and update transit
		double sum0 = countTransit[0][0] + countTransit[0][1];
		double sum1 = countTransit[1][0] + countTransit[1][1];
		transit[0][0] = countTransit[0][0] / sum0;
		transit[0][1] = countTransit[0][1] / sum0;
		transit[1][0] = countTransit[1][0] / sum1;
		transit[1][1] = countTransit[1][1] / sum1;

		//update counter
		count++;

		//report
		resultFile << transit[0][0] << " " << transit[1][0] << " " << lambda0 << " " << lambda1 << std::endl;
	}
}



/*
double derivative(double x, std::function<double(double)> f) {
	const double epsilon = 0.000001;
	return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon);
}*/

double gammaIteration(double alpha, double x, double lnx) { //one iteration for the numerical calculation in MLE of gamma distribution
	//double rhs = 1.0 / alpha + (lnx - log(x) + log(alpha) - boost::math::digamma(alpha)) / (alpha - alpha * boost::math::polygamma(1, alpha));
	//return 1.0 / rhs;
	return std::max(0.0001, alpha - (log(alpha) - boost::math::digamma(alpha) - log(x) + lnx) / (1 / alpha - boost::math::polygamma(1, alpha))); //non-zero constraint
}

void gammaEstimator(double x, double lnx, double& alpha, double& beta) { // mean of x, and mean of lnx; natural sufficient statistics.
	//parameters
	double hoge = log(x) - lnx;
	const int iterationNo = 150;  //maximum iteration no
	const int epsilon = 1.0e-6;  //if the difference of original alpha and updated alpha is smaller than this number, then iteration halts;

	//initial value
	//alpha = 0.5 / (log(x) - lnx);


	//numerical calculate alpha
	for (int i = 0; i != iterationNo; i++) {
		double oldAlpha = alpha;
		alpha = gammaIteration(alpha, x, lnx);
		if (abs(oldAlpha - alpha) < epsilon)
			break;		
	}

	//cut-off
	//variance = x^2 / alpha
	//const double minVariance = 100.0;
	//alpha = std::min(alpha, x*x / minVariance);

	//then, calculate beta
	beta = alpha / x;
}

double Tree2LogSurvivalTime(const Node& n, int i, const std::vector<std::vector<double>>& integrated, double dt, double max, std::vector<double>& infiniteVal, const std::vector<Density*>& distributions, bool leafIncluded = true) { //for M step in tree2Estimator, integrated[i] stores \int_0^t x p(x) dx with step size dt until t = max for type i.  InfiniteVal = \int_0^\infty x f(x) dx
	if (leafIncluded && (n.isLeaf() || n.isroot())) { //root and leaves are treated separetely
		//return 0.0;
		int iterator = int(((std::max(std::min(max, n.survivalTime), 0.1)) - 0.1) / dt);// indicate appropriate position of integrated
		// for debug
		//double hoget = (infiniteVal[i] - integrated[i][iterator]) / (1.0 - distributions[i]->CDF(n.survivalTime));
		return n.currentDistribution[i] * (infiniteVal[i] - integrated[i][iterator]) / (1.0 - distributions[i]->CDF(n.survivalTime));
	}
	else if (n.isLeaf() || n.isroot())
		return 0.0;
	else
		return n.currentDistribution[i] * log(n.survivalTime);
}

double Tree2SurvivalTime(const Node& n, int i, const std::vector<std::vector<double>>& integrated, double dt, double max, const std::vector<double>& infiniteVal, const std::vector<Density*>& distributions, bool leafIncluded = true) { //for M step in tree2Estimator
	if (leafIncluded && (n.isLeaf() || n.isroot())) { //root and leaves are treated separetely
		//return 0.0;
		int iterator = int(std::min(max, n.survivalTime) / dt);// indicate appropriate position of integrated
		return n.currentDistribution[i] * (infiniteVal[i] - integrated[i][iterator]) / (1.0 - distributions[i]->CDF(n.survivalTime));
	}
	else if (n.isLeaf() || n.isroot())
		return 0.0;
	else
		return n.currentDistribution[i] * n.survivalTime;
}

double Tree2weight(const Node& n, int i, bool leafIncluded = true) { //for M step in tree2Estimator
	if (!leafIncluded && ( n.isLeaf() || n.isroot()))
		return 0.0;
	return n.currentDistribution[i];
}

void integrate(std::vector<double>& res, const std::function<double(double)>& f, double dt, double min, double max) { // store \int_min^t f(x) dx with step size dt for all t in [min, max];
	assert(dt > 0);
	assert(min < max);

	double temp = 0; //store current integrated value
	double t = min;

	res.push_back(0.0); // for t = min

	//apply simpson's fomula iteratively
	while (t < max) {
		//simpcon's fomula to caclulate \int_min^(t + dt) f(x) dx
		temp += dt / 6.0 * (f(t) + 4 * f(t + dt / 2.0) + f(t + dt));

		//store the result
		res.push_back(temp);

		//update t
		t += dt;
	}
}


void tree2EstimatorCore(std::vector<Tree*>& forest, std::vector<std::vector<double>>& transit, double& alpha0, double& alpha1, double& beta0, double& beta1, std::ofstream& resultFile, bool leafIncluded = true, bool weightingCorrection = false) {
	//Stop condition
	const int maxIteration = 300;
	const double epsilon = 1.0e-7;
	//integration parameters
	const double dt = 0.01;
	const double integrationMax = 400.0;

	//report total node, leaf node
	int nodesNo, leavesNo;
	nodesNo = leavesNo = 0;
	for (auto itr = forest.begin(); itr != forest.end(); itr++) {
		nodesNo += (*itr)->size();
		leavesNo += (*itr)->leavesSize();
	}
	std::cout << "No. of nodes is: " << nodesNo << std::endl;
	std::cout << "No. of leaves is: " << leavesNo << std::endl;

	//for iteration
	int count = 0; //count # iterations
	std::vector<Density*> distributions(2, NULL);

	//for stop condition
	double logLikeold = -1.0e10;

	//iteration
	while (count < maxIteration) {
		//initialize distributions
		GammaDensity dist0(alpha0, beta0), dist1(alpha1, beta1);
		distributions[0] = &dist0; distributions[1] = &dist1;

		//E-step
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {
			(*itr)->inference(transit, distributions);
		}

		//report log-likelihood
		double logLik = 0;
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {
			logLik += (*itr)->logLik();
		}
		if (count % 50 == 0)
			std::cout << "The log-Like after " << count << "-th iteration is: " << logLik << std::endl;
		//stop condition on log likelyhood
		if (logLik - logLikeold < epsilon)
			break;
		else
			logLikeold = logLik;
		resultFile << logLik << " ";

		//M-step
		//for alpha and beta
		double x0, x1, lnx0, lnx1, weight0, weight1;  //estimate E[x] and E[ln x]. x and lnx are weighted-sums. normx and normLnx are normalization factors. 
		x0 = x1 = lnx0 = lnx1 = weight0 = weight1 = 0.0;


		//separated summation for leaf and root note;
		//first do integration
		std::vector<std::vector<double>> xint(2, std::vector<double>()), logint(2, std::vector<double>()); //\int x*p(x) dx, \int logx * p(x) dx; suffix indicatets the type
		integrate(xint[0], [&](double x) {return distributions[0]->density(x) * x; }, dt, 0, integrationMax);
		integrate(xint[1], [&](double x) {return distributions[1]->density(x) * x; }, dt, 0, integrationMax);
		integrate(logint[0], [&](double x) {return distributions[0]->density(x) * log(x); }, dt, 0.1, integrationMax);
		integrate(logint[1], [&](double x) {return distributions[1]->density(x) * log(x); }, dt, 0.1, integrationMax);

		//Set InfiniteValue i.e. \int_0^\infty f(x)p(x) dx
		std::vector<double> inftyValx, inftyValLog;
		inftyValx.push_back(alpha0 / beta0); inftyValx.push_back(alpha1 / beta1); 
		inftyValLog.push_back(boost::math::digamma(alpha0) - log(beta0)); 
		inftyValLog.push_back(boost::math::digamma(alpha1) - log(beta1));

		//sum up 
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {
			std::vector<double> sum1, sum2, sum3;
			double lambda = forest[0]->lambda;
			if (weightingCorrection) {
				sum1 = (*itr)->sum([&](const Node& n, int i) {return exp(lambda * n.survivalTime) * Tree2LogSurvivalTime(n, i, logint, dt, integrationMax, inftyValLog, distributions, leafIncluded); });
				sum2 = (*itr)->sum([&](const Node& n, int i) {return exp(lambda * n.survivalTime) *Tree2SurvivalTime(n, i, xint, dt, integrationMax, inftyValx, distributions, leafIncluded); });
				sum3 = (*itr)->sum([&](const Node& n, int i) {return exp(lambda * n.survivalTime) *Tree2weight(n, i, leafIncluded); });
			}
			else {
				sum1 = (*itr)->sum([&](const Node& n, int i) {return Tree2LogSurvivalTime(n, i, logint, dt, integrationMax, inftyValLog, distributions, leafIncluded); });
				sum2 = (*itr)->sum([&](const Node& n, int i) {return Tree2SurvivalTime(n, i, xint, dt, integrationMax, inftyValx, distributions, leafIncluded); });
				sum3 = (*itr)->sum([&](const Node& n, int i) {return Tree2weight(n, i, leafIncluded); });
			}
			x0 += sum2[0]; x1 += sum2[1];
			lnx0 += sum1[0]; lnx1 += sum1[1];
			weight0 += sum3[0]; weight1 += sum3[1];
		}



		//M step for transition Matrix
		std::vector<std::vector<double>> countTransit(2, std::vector<double>(2, 0.0));
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {//sum up for all trees
			std::vector<std::vector<double>> thisTreeCount = (*itr)->transitCoef(transit, distributions);
			countTransit[0][0] += thisTreeCount[0][0];
			countTransit[0][1] += thisTreeCount[0][1];
			countTransit[1][0] += thisTreeCount[1][0];
			countTransit[1][1] += thisTreeCount[1][1];
		}

		//update parameters
		//alpha and gamma
		gammaEstimator(x0 / weight0, lnx0 / weight0, alpha0, beta0);
		gammaEstimator(x1 / weight1, lnx1 / weight1, alpha1, beta1);
		//normalize and update transit
		double sum0 = countTransit[0][0] + countTransit[0][1];
		double sum1 = countTransit[1][0] + countTransit[1][1];
		transit[0][0] = countTransit[0][0] / sum0;
		transit[0][1] = countTransit[0][1] / sum0;
		transit[1][0] = countTransit[1][0] / sum1;
		transit[1][1] = countTransit[1][1] / sum1;

		//update counter
		count++;

		//report
		resultFile << transit[0][0] << " " << transit[1][0] << " " << alpha0 << " " << beta0 << " " << alpha1 << " " << beta1 << std::endl;
	}

}

double midPoint(double eta, double neo, double old) {
	return (1.0 - eta) * old + neo * eta;
}

void WakamotoEstimatorCore(std::vector<Tree*>& forest, std::vector<std::vector<double>>& transit, std::vector<double>& alpha, std::vector<double>& beta, double maxSurvival, std::ofstream& resultFile, bool leafIncluded = true, bool weightingCorrection = false) {
	//Stop condition
	const int maxIteration = 300;
	const double epsilon = 1.0e-5;
	const double eta = 1.0; //learning parameter
	//integration parameters
	const double dt = 0.01;
	const double integrationMax = maxSurvival;

	//typeno
	assert(!forest.empty());
	int typeno = forest[0]->typeNo();
	assert(alpha.size() == typeno && beta.size() == typeno);

	//report total node, leaf node
	int nodesNo, leavesNo;
	nodesNo = leavesNo = 0;
	for (auto itr = forest.begin(); itr != forest.end(); itr++) {
		nodesNo += (*itr)->size();
		leavesNo += (*itr)->leavesSize();
	}
	std::cout << "No. of nodes is: " << nodesNo << std::endl;
	std::cout << "No. of leaves is: " << leavesNo << std::endl;

	//for iteration
	int count = 0; //count # iterations
	std::vector<Density*> distributions(typeno, NULL);

	//for stop condition
	double logLikeold = -1.0e-10;

	//iteration
	while (count < maxIteration) {
		//initialize distributions
		//GammaDensity dist0(alpha0, beta0), dist1(alpha1, beta1);
		//distributions[0] = &dist0; distributions[1] = &dist1;

		for (int i = 0; i != typeno; i++) {
			distributions[i] = new GammaDensity(alpha[i], beta[i]);
		}

		//E-step
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {
			(*itr)->inference(transit, distributions);
		}

		//report log-likelihood
		double logLik = 0;
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {
			logLik += (*itr)->logLik(leafIncluded);
		}
		if (count % 50 == 0)
			std::cout << "The log-Like after " << count << "-th iteration is: " << logLik << std::endl;
		//stop condition on log likelyhood
		if (abs(logLik - logLikeold) < epsilon || (logLik - logLikeold < 0 && count > 10))
			break;
		else
			logLikeold = logLik;
		resultFile << logLik << " ";

		//M-step
		//for alpha and beta
		std::vector<double> Ex(typeno, 0), Elnx(typeno, 0), E1(typeno, 0); //E[x], E[lnx] E[1] of emperical process. index is type


		//separated summation for leaf and root note;
		//first do integration
		std::vector<std::vector<double>> xint(typeno, std::vector<double>()), logint(typeno, std::vector<double>()); //\int x*p(x) dx, \int logx * p(x) dx; suffix indicatets the type
		for (int i = 0; i != typeno; i++) {
			integrate(xint[i], [&](double x) {return distributions[i]->density(x) * x; }, dt, 0.1, integrationMax);
			integrate(logint[i], [&](double x) {return distributions[i]->density(x) * log(x); }, dt, 0.1, integrationMax);
		}

		//Set InfiniteValue i.e. \int_0^\infty f(x)p(x) dx
		std::vector<double> inftyValx, inftyValLog;
		for (int i = 0; i != typeno; i++) {
			inftyValx.push_back(alpha[i] / beta[i]); 
			inftyValLog.push_back(boost::math::digamma(alpha[i]) - log(beta[i]));
		}

		//sum up 
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {
			double lambda = forest[0]->lambda;
			std::vector<double> sum1, sum2, sum3;
			if(weightingCorrection){
				sum1 = (*itr)->sum([&](const Node& n, int i) {return exp(lambda * n.survivalTime) * Tree2LogSurvivalTime(n, i, logint, dt, integrationMax, inftyValLog, distributions, leafIncluded); });
				sum2 = (*itr)->sum([&](const Node& n, int i) {return exp(lambda * n.survivalTime) *Tree2SurvivalTime(n, i, xint, dt, integrationMax, inftyValx, distributions, leafIncluded); });
				sum3 = (*itr)->sum([&](const Node& n, int i) {return exp(lambda * n.survivalTime) *Tree2weight(n, i, leafIncluded); });
			}
			else {
				sum1 = (*itr)->sum([&](const Node& n, int i) {return Tree2LogSurvivalTime(n, i, logint, dt, integrationMax, inftyValLog, distributions, leafIncluded); });
				sum2 = (*itr)->sum([&](const Node& n, int i) {return Tree2SurvivalTime(n, i, xint, dt, integrationMax, inftyValx, distributions, leafIncluded); });
				sum3 = (*itr)->sum([&](const Node& n, int i) {return Tree2weight(n, i, leafIncluded); });
			}
			//x0 += sum2[0]; x1 += sum2[1];
			//lnx0 += sum1[0]; lnx1 += sum1[1];
			//weight0 += sum3[0]; weight1 += sum3[1];
			for (int i = 0; i != typeno; i++) {
				Elnx[i] += sum1[i];
				Ex[i] += sum2[i];
				E1[i] += sum3[i];
			}
		}



		//M step for transition Matrix
		std::vector<std::vector<double>> countTransit(typeno, std::vector<double>(typeno, 0.0));
		for (auto itr = forest.begin(); itr != forest.end(); itr++) {//sum up for all trees
			std::vector<std::vector<double>> thisTreeCount = (*itr)->transitCoef(transit, distributions);
			//countTransit[1][1] += thisTreeCount[1][1];
			for (int i = 0; i != typeno; i++) {
				for (int j = 0; j != typeno; j++)
					countTransit[i][j] += thisTreeCount[i][j];
			}
		}

		//update parameters
		//alpha and gamma
		for (int i = 0; i != typeno; i++) {
			double tempAlpha = alpha[i];
			double tempBeta = beta[i];
			gammaEstimator(Ex[i] / E1[i], Elnx[i] / E1[i], tempAlpha, tempBeta);
			alpha[i] = midPoint(eta, tempAlpha, alpha[i]);
			beta[i] = midPoint(eta, tempBeta, beta[i]);
		}
		//normalize and update transit
		std::vector<double> sum(typeno, 0);
		for (int i = 0; i != typeno; i++) {
			for (int j = 0; j != typeno; j++) {
				sum[i] += countTransit[i][j];
			}
		}
		//transit[0][0] = countTransit[0][0] / sum0;
		//transit[0][1] = countTransit[0][1] / sum0;
		//transit[1][0] = countTransit[1][0] / sum1;
		//transit[1][1] = countTransit[1][1] / sum1;
		for (int i = 0; i != typeno; i++) {
			for (int j = 0; j != typeno; j++) {
				//transit[i][j] = countTransit[i][j] / sum[i];
				if (sum[i] > epsilon)
					transit[i][j] = midPoint(eta, countTransit[i][j] / sum[i], transit[i][j]);
				else
					transit[i][j] = 1.0 / typeno;
			}
		}

		
		//free memory
		for (int i = 0; i != typeno; i++)
			delete distributions[i];

		//update counter
		count++;

		//report
		//transit
		for (int i = 0; i != typeno; i++) {
			for (int j = 0; j != typeno; j++)
				resultFile << transit[i][j] << " ";
		}
		//Parameter for gamma distribution
		for (int i = 0; i != typeno; i++)
			resultFile << alpha[i] << " " << beta[i] <<" ";
		//fraction of each type
		for (int i = 0; i != typeno; i++)
			resultFile << E1[i] / nodesNo << " ";
		resultFile << std::endl;
	}

}