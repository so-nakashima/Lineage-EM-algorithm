#include "Tree.h"
#include <string>
#include <vector>
#include "Density.h"
#include <queue>
#include <iostream>
#include <fstream>
#include "estimator.h"
#include <string>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>

void tree1Estimator() { //em extimator for tree 1 (see also note 19 june may)
						//meta-parameters
						// store the trees' id which will be loaded
	std::vector<int> loadNo;
	//loadNo.push_back(4);
	for (int i = 0; i != 10; i++) {//modify as you like
		loadNo.push_back(i);
	}
	//estimated steady distribution and lambda
	const double lambda = .0365397;
	const std::vector<double> steadyDistribution{ .471824 , .528176 };
	//initial parameter
	double lambda0, lambda1;
	lambda0 = lambda1 = lambda; //parameter for ExponentialDensity distribution
								//lambda0 = 0.048; lambda1 = 0.028;
	std::vector<std::vector<double>> transit(2, std::vector<double>(2, 0)); //parameter for transit matrix
	transit[0][0] = steadyDistribution[0];
	transit[0][1] = steadyDistribution[1];
	transit[1][0] = steadyDistribution[0];
	transit[1][1] = steadyDistribution[1];
	/*transit[0][0] = 0.8;
	transit[0][1] = 0.2;
	transit[1][0] = 0.4;
	transit[1][1] = 0.6;*/
	//result file
	std::ofstream resultFile("resTree1.txt");

	//load tree data
	std::vector<Tree*> forest; //Memory leak
	std::string namebase = ".\\Tree1_120min\\Tree1_";
	for (auto itr = loadNo.begin(); itr != loadNo.end(); itr++) {
		Tree* temp = new Tree(steadyDistribution, lambda);
		temp->load(namebase + std::to_string(*itr) + ".dat");
		if (temp->size() != 1) // dont use only root node tree
			forest.push_back(temp);
	}

	tree1EstimatorCore(forest, transit, lambda0, lambda1, resultFile);

	//output
	printf("estimation end\n lambda0: %f\n lambda1: %f\n T[0][0] = %f\n, T[1][0] = %f\n", lambda0, lambda1, transit[0][0], transit[1][0]);
}


void tree2Estimator() { //em extimator for tree 2 (see also note 25 june may)
						//meta-parameters
						// store the trees' id which will be loaded
	std::vector<int> loadNo;
	//loadNo.push_back(4);
	for (int i = 0; i != 1; i++) {//modify as you like
		loadNo.push_back(i);
	}
	//estimated steady distribution and lambda
	const double lambda = .028722;
	const std::vector<double> steadyDistribution{ .476386 , .523614 };
	//initial parameter
	double alpha0, alpha1, beta0, beta1;
	//lambda0 = lambda1 =  lambda; //parameter for GammaDensity distribution
	alpha0 = 12.73475736217195; beta0 = 0.57295742743472433; alpha1 = 12.73475736217195; beta1 = 0.57295742743472433;
	std::vector<std::vector<double>> transit(2, std::vector<double>(2, 0)); //parameter for transit matrix
	//transit[0][0] = steadyDistribution[0];
	//transit[0][1] = steadyDistribution[1];
	//transit[1][0] = steadyDistribution[0];
	//transit[1][1] = steadyDistribution[1];
	transit[0][0] = 0.5;
	transit[0][1] = 0.5;
	transit[1][0] = 0.5;
	transit[1][1] = 0.5;
	//result file
	std::ofstream resultFile(".\\paper\\res\\Tree2_withLeafInfo.txt");

	//load tree data
	std::vector<Tree*> forest; //Memory leak
	std::string namebase = ".\\paper\\tree\\Tree2_220min_";
	for (auto itr = loadNo.begin(); itr != loadNo.end(); itr++) {
		Tree* temp = new Tree(steadyDistribution, lambda);
		temp->load(namebase + std::to_string(*itr) + ".txt");  //change Wakamoto-type or Hormoz-type
		if (temp->size() != 1)  // dont use only root node tree
			forest.push_back(temp);
	}

	tree2EstimatorCore(forest, transit, alpha0, alpha1, beta0, beta1, resultFile);
	//WakamotoEstimatorCore(forest, transit, alpha0, alpha1, beta0, beta1, resultFile);

	forest[0]->MathematicaGraphics(".\\paper\\res\\estimatedStatesTree2WithLeaveInfo.txt");
}

void tree1Stat() {
	int treeno = 20;
	int avgno = 500;
	std::vector<std::vector<double>> samples(4);
	std::ofstream out1(".\\res\\Tree1sample.txt");
	std::ofstream out2(".\\res\\Tree1sampleStats.txt");

	for (int count = 0; count != avgno; count++) {
		std::vector<int> loadNo;
		//loadNo.push_back(4);
		for (int i = count*treeno; i != (count + 1) * treeno; i++) {//modify as you like
			loadNo.push_back(i);
		}
		//estimated steady distribution and lambda
		const double lambda = .0365397;
		const std::vector<double> steadyDistribution{ .471824 , .528176 };
		//initial parameter
		double lambda0, lambda1;
		lambda0 = lambda1 = lambda; //parameter for ExponentialDensity distribution
		std::vector<std::vector<double>> transit(2, std::vector<double>(2, 0)); //parameter for transit matrix
		transit[0][0] = steadyDistribution[0];
		transit[0][1] = steadyDistribution[1];
		transit[1][0] = steadyDistribution[0];
		transit[1][1] = steadyDistribution[1];
		/*transit[0][0] = 0.8;
		transit[0][1] = 0.2;
		transit[1][0] = 0.4;
		transit[1][1] = 0.6;*/
		//result file
		std::ofstream resultFile("resTree1.txt");

		//load tree data
		std::vector<Tree*> forest; //Memory leak
		std::string namebase = ".\\Tree1_120min\\Tree1_";
		for (auto itr = loadNo.begin(); itr != loadNo.end(); itr++) {
			Tree* temp = new Tree(steadyDistribution, lambda);
			temp->load(namebase + std::to_string(*itr) + ".dat");
			if (temp->size() != 1) // dont use only root node tree
				forest.push_back(temp);
		}

		tree1EstimatorCore(forest, transit, lambda0, lambda1, resultFile);

		//output
		out1 << transit[0][0] << " " << transit[1][0] << " " << lambda0 << " " <<  lambda1 << std::endl;
		samples[0].push_back(transit[0][0]);
		samples[1].push_back(transit[1][0]);
		samples[2].push_back(lambda0);
		samples[3].push_back(lambda1);
	}

	for (int i = 0; i != 4; i++) {
		boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::variance, boost::accumulators::tag::mean>> acc;
		boost::for_each(samples[i], boost::bind<void>(boost::ref(acc), _1));

		out2 << boost::accumulators::extract::mean(acc) << " " << boost::accumulators::extract::variance(acc) << std::endl;
	}
}

void tree2Stat() {
	int treeno = 10;
	int avgno = 100;
	std::vector<std::vector<double>> samples(6);
	std::ofstream out1(".\\paper\\Tree4.txt");
	std::ofstream out2(".\\paper\\Tree4stats.txt");

	for (int i = 0; i != avgno; i++) {
		std::vector<int> loadNo;
		//loadNo.push_back(4);
		for (int j = i*treeno; j != (i+1)*treeno; j++) {
			loadNo.push_back(i);
		}

		//estimated steady distribution and lambda
		const double lambda = .028722;
		const std::vector<double> steadyDistribution{ .3 , .7 };
		//initial parameter
		double alpha0, alpha1, beta0, beta1;
		//lambda0 = lambda1 =  lambda; //parameter for GammaDensity distribution
		//std::vector<double> alpha(2, 12.73475736217195), beta(2, 0.57295742743472433);

		std::vector<double> alpha(2, 15.0), beta(2, 0.11);  //tree4
		alpha[0] = 5.0; //tree4

		alpha0 = 7; beta0 = .1; alpha1 = 20; beta1 = 0.1;
		std::vector<std::vector<double>> transit(2, std::vector<double>(2, 0)); //parameter for transit matrix
		//transit[0][0] = steadyDistribution[0] + 0.01;
		//transit[0][1] = steadyDistribution[1] - 0.01;
		//transit[1][0] = steadyDistribution[0];
		//transit[1][1] = steadyDistribution[1];
		transit[0][0] = .6;
		transit[0][1] = .4;
		transit[1][0] = .4;
		transit[1][1] = .6;
		//result file
		std::ofstream resultFile(".\\res\\resTree2_220min.txt");

		//load tree data
		std::vector<Tree*> forest; //Memory leak
		std::string namebase = ".\\paper\\tree\\Tree4\\Tree4_250min_";
		for (auto itr = loadNo.begin(); itr != loadNo.end(); itr++) {
			Tree* temp = new Tree(steadyDistribution, lambda);
			temp->load(2, namebase + std::to_string(*itr) + ".txt", true);
			if (temp->size() != 1)  // dont use only root node tree
				forest.push_back(temp);
		}

		//WakamotoEstimatorCore(forest, transit, alpha, beta, 200, resultFile, false, true);

		//


		//samples[2].push_back(alpha[0]);
		//samples[3].push_back(beta[0]);
		//samples[4].push_back(alpha[1]);
		//samples[5].push_back(beta[1]);


		//
		//out1 << transit[0][0] << " " << transit[1][0] << " " << alpha[0] << " " << alpha[1] << " " << beta[0] << " " << beta[1] << " " << std::endl;


		tree2EstimatorCore(forest, transit, alpha0, alpha1, beta0, beta1, resultFile, false, false);
		samples[2].push_back(alpha0);
		samples[3].push_back(beta0);
		samples[4].push_back(alpha1);
		samples[5].push_back(beta1);
		out1 << transit[0][0] << " " << transit[1][0] << " " << alpha0 << " " << alpha1 <<  " " << beta0 << " " << beta1 << " " << std::endl;



		samples[0].push_back(transit[0][0]);
		samples[1].push_back(transit[1][0]);
	}

	for (int i = 0; i != 6; i++) {
		boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::variance, boost::accumulators::tag::mean>> acc;
		boost::for_each(samples[i], boost::bind<void>(boost::ref(acc), _1));

		out2 << boost::accumulators::extract::mean(acc) << " " << boost::accumulators::extract::variance(acc) << std::endl;
	}
}



void pushWakamotoTrees(std::vector<Tree*>& forest, std::string namebase, int treeNo, int typeno, double lambda) {  //Be aware, this function allocates memory for trees in forest
	//steady distribution
	std::vector<double> stationaryDistribution(typeno, 1.0 / typeno);

	for (int i = 0; i != treeNo; i++) {
		Tree* t = new Tree(stationaryDistribution, lambda);
		std::string filename = namebase + std::to_string(i) + ".txt";
		t->load(typeno, filename);
		if(t->size() > 1)
			forest.push_back(t);
	}
}

void ReadWakamoto(std::string experiment, std::vector<Tree*>& forest, int typeno, double& alpha, double& beta, double& maxSurvival) {
	if (experiment == "F3_rpsL_Glc_30C_3min") {
		pushWakamotoTrees(forest, ".\\RawData\\F3_rpsL_Glc_30C_3min\\Results0013.xls", 22, typeno, 1.0);
		pushWakamotoTrees(forest, ".\\RawData\\F3_rpsL_Glc_30C_3min\\Results0072.xls", 24, typeno, 1.0);
		pushWakamotoTrees(forest, ".\\RawData\\F3_rpsL_Glc_30C_3min\\Results0253.xls", 35, typeno, 1.0);
		pushWakamotoTrees(forest, ".\\RawData\\F3_rpsL_Glc_30C_3min\\Results0284.xls", 23, typeno, 1.0);
		
		//MLE parameter
		//alpha = 4.385057886354065;
		//beta = 0.047898657198265727;
		//maxSurvival = 800;
	}
	if (experiment == "toy") {
		//steady distribution
		std::vector<double> stationaryDistribution(typeno, 1.0 / typeno);

		Tree* t = new Tree(stationaryDistribution, .028722);
		t->load(typeno, ".\\paper\\tree\\Tree2_220min_nonLeave0.txt");
		forest.push_back(t);

		alpha = 12.73475736217195;
		beta = 0.57295742743472433;
		maxSurvival = 200;
	}
	else {
		std::cout << "Such Experiment does NOT exist..." << std::endl;
		abort();
	}
}

void WakamotoEstimator(std::string experiment, std::string resultFile, int typeno) {
	std::vector<Tree*> forest;
	double MLEalpha, MLEbeta, maxSurvival;

	//Read
	ReadWakamoto(experiment, forest, typeno, MLEalpha, MLEbeta, maxSurvival);

	//InitialValue
	std::vector<std::vector<double>> transit(typeno, std::vector<double>(typeno, 0));
	for (int i = 0; i != typeno; i++) {
		for (int j = 0; j != typeno; j++) {
			transit[i][j] = 1.0 / typeno;
		}
	}

	std::vector<double> alpha(typeno, MLEalpha), beta(typeno, MLEbeta);
	//alpha[0] = 20; beta[0] = 1.0;
	//alpha[1] = 10; beta[0] = .05;
	alpha[0] += .1;
	alpha[1] -= .1;
	//alpha[2] = 10.0;
	//beta[2] = 10.0 / 30;

	//resultFile
	std::ofstream file(resultFile);

	//Execute
	WakamotoEstimatorCore(forest, transit, alpha, beta, maxSurvival, file, true);

	forest[0]->MathematicaGraphics(".\\paper\\res\\estimatedStatesTree2WithoutLeaveInfo.txt");
}