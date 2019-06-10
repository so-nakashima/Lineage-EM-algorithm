#include "Tree.h"
#include <iostream>
#include <vector>
#include <random>
#include "estimator.h"
#include "test.h"
#include "TreeLDS.h"
#include <fstream>
#include <Eigen/Core>
#include "dataGenerator.h"
#include "LDSestimator.h"
#include "execution-estimator.h"
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

using namespace Eigen;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;

void inferenceTest() {
	std::vector<double> steady;
	steady.push_back(0.5); steady.push_back(0.5);

	std::vector<std::vector<double>> transit(2, std::vector<double>(2, 0));
	transit[0][0] = transit[1][1] = 0.7;
	transit[0][1] = transit[1][0] = 0.3;

	double lambda0, lambda1;
	lambda0 = 0.048; lambda1 = 0.028;
	std::vector<Density*> distributions(2, NULL);
	ExponentialDensity dist0(lambda0), dist1(lambda1);
	distributions[0] = &dist0; distributions[1] = &dist1;

	Tree t;
	t.load(".\\Tree1_120min\\Tree1_8.dat");
	t.steadyDistribution = steady;
	t.inference(transit, distributions);
	
	int hoge = 0;
}

double f(Node n, int i) { // for pair inference test
	return n.survivalTime;
}


void pairInferenceTest() {
	std::vector<double> steady;
	steady.push_back(0.5); steady.push_back(0.5);

	std::vector<std::vector<double>> transit(2, std::vector<double>(2, 0));
	transit[0][0] = transit[1][1] = 0.7;
	transit[0][1] = transit[1][0] = 0.3;

	Density* noage = new Density;
	std::vector<Density*> densities(2, noage);
	Tree t(steady, 2.0);
	t.load(".\\Tree1\\Tree1_1.dat");
	std::vector<double> hoge = t.sum(f);
	t.inference(transit, densities);
	std::vector<std::vector<double>> fuga = t.transitCoef(transit, densities);
}

void integrationTest() {
	std::vector<double> hoge;
	integrate(hoge, [](double x) {return x*x; }, 0.1, 0, 10.0);
}

void gammaEstimationTest() {
	double alpha_t = 200.0;
	double beta_t = 1.0;
	double alpha = 1.0;
	double beta = 102.0;
	gammaEstimator(alpha_t / beta_t, boost::math::digamma(alpha_t) - log(beta_t), alpha, beta);
}

void temporalTest(){
	TreeLDS t;
	t.load("testTreeLDS.txt", 2, 2);

	return;
}

void LDSinferenceTest() {
	int generation = 10;

	//int n = 2;
	//int m = 2;
	//Mat A(n, n);
	//A << 1, 0, 0, 1;
	//Mat C(m, n);
	//C << 1, 0, 0, 1;
	//Mat sigmaW(n, n);
	//sigmaW << .5, 0, 0, .5;
	//Mat sigmaV(m, m);
	//sigmaV << .5, 0, 0, .5;
	//Vec initState(2);
	//initState << 1, 0;

	//tree2

	//int n = 2;
	//int m = 2;
	//double theta = 3.1415 / 3.0;
	//Mat A(n, n);
	//A << cos(theta), -sin(theta), sin(theta), cos(theta);
	//Mat C(m, n);
	//C << 1, 0, 0, 1;
	//Mat sigmaW(n, n);
	//sigmaW << .0001, 0, 0, .0001;
	//Mat sigmaV(m, m);
	//sigmaV << .0001, 0, 0, .0001;
	//Vec initState(2);
	//initState << 1, 0;

	//tree 3
	//int n = 2;
	//int m = 1;
	//double theta = 3.1415 / 3.0;
	//Mat A(n, n);
	//A << 1, 1, 0, 1;
	//Mat C(m, n);
	//C << 1, 0;
	//Mat sigmaW(n, n);
	//sigmaW << .001, 0, 0, .001;
	//Mat sigmaV(m, m);
	//sigmaV << .1;
	//Vec initState(2);
	//initState << 1, 1;

	//tree4
	int n = 2;
	int m = 2;
	Mat A(n, n);
	A << 1, 1, 0, 1;
	Mat C(m, n);
	C << 1, 0, 0, 1;
	Mat sigmaW(n, n);
	sigmaW << .1, 0, 0, .1;
	Mat sigmaV(m, m);
	sigmaV << .1, 0, 0, .1;
	Vec initState(2);
	initState << 1, 1;

	//tree5

	//int n = 1;
	//int m = 1;
	//Mat A(n, n);
	//A << 1;
	//Mat C(m, n);
	//C << 1;
	//Mat sigmaW(n, n);
	//sigmaW << .1;
	//Mat sigmaV(m, m);
	//sigmaV << .1;
	//Vec initState(n);
	//initState << 1;

	//output
	std::ofstream out(".\\LDS1\\resLDSdebug2.txt");

	Mat rootVar(n, n);
	rootVar << 0.00001, 0, 0, 0.00001;
	//rootVar << 0.00001;
	Vec rootMean(n);
	rootMean = initState;

	//datageneration
	std::string filename = ".\\LDS1\\tree2_";
	int i = 0;
	std::ofstream outTree(filename + std::to_string(i) + ".txt");
	std::ofstream trueState(filename + std::to_string(i) + "_true.txt");
	int current = 0;
	generateOneLDSTree(n, m, current, A, C, sigmaW, sigmaV, generation, -1, initState, outTree, trueState, [] {return 2; });

	//prediction
	TreeLDS t;
	t.load(".\\LDS1\\tree2_0.txt", n, m);
	//t.load(".\\LDS1\\tree2_ex2.txt", n, m);

	
	//Mat a(n,n);
	//a << 1, 0, 0, 1;
	//Mat sigmaw(n, n);
	//sigmaw , 0, 0, 10000;

	Mat a(n,n); a << 0.501276, -0.915556, 0.8482, 0.498875;
	Mat c(m,n); c << -0.00300284, -0.173368, 0.169364, -0.00213864;
	Mat w(n,n); w << 0.00891825, -0.00469345, -0.00469345, 0.01412;
	Mat v(m,m); v << 0.000513069, 0.000209337, 0.000209337, 0.000388186;
	Vec rmean(n); rmean << -0.468277, -5.23107;
	Mat rvar(n, n); rvar << 0.00160193, 0.000291012, 0.000291012, 0.00205969;
	
	t.smoothing(A, C, sigmaW, sigmaV, rootMean, rootVar);
	//t.smoothing(a, c, w, v, rmean, rvar);

	t.output(out);
	std::cout << t.logLik() << std::endl;
}

void LDSestimateTest() {
	int generation = 15;

	//int n = 2;
	//int m = 2;
	//Mat A(n, n);
	//A << 1, 0, 0, 1;
	//Mat C(m, n);
	//C << 1, 0, 0, 1;
	//Mat sigmaW(n, n);
	//sigmaW << .5, 0, 0, .5;
	//Mat sigmaV(m, m);
	//sigmaV << .5, 0, 0, .5;
	//Vec initState(2);
	//initState << 1, 1;

	//tree2

	//int n = 2;
	//int m = 2;
	//double theta = 3.1415 / 3.0;
	//Mat A(n, n);
	//A << cos(theta), -sin(theta), sin(theta), cos(theta);
	//Mat C(m, n);
	//C << 1, 0, 0, 1;
	//Mat sigmaW(n, n);
	//sigmaW << .1, 0, 0, .1;
	//Mat sigmaV(m, m);
	//sigmaV << .1, 0, 0, .1;
	//Vec initState(2);
	//initState << 1, 0;

	//tree 3
	//int n = 2;
	//int m = 1;
	//double theta = 3.1415 / 3.0;
	//Mat A(n, n);
	//A << 1, 1, 0, 1;
	//Mat C(m, n);
	//C << 1, 0;
	//Mat sigmaW(n, n);
	//sigmaW << .1, 0, 0, .1;
	//Mat sigmaV(m, m);
	//sigmaV << .1;
	//Vec initState(2);
	//initState << 1, 1;

	//tree5
	//int n = 1;
	//int m = 1;
	//Mat A(n, n);
	//A << 1;
	//Mat C(m, n);
	//C << 1;
	//Mat sigmaW(n, n);
	//sigmaW << .1;
	//Mat sigmaV(m, m);
	//sigmaV << .1;
	//Vec initState(2);
	//initState << 1;


	//tree 6
	int n = 3;
	int m = 1;
	Mat A(n, n);
	A << 1, 0, 0, 0, 1, 0, 0, 0, 1;
	Mat C(m, n);
	C << 1, 0, 0;
	Mat sigmaW(n, n);
	sigmaW << .1, 0, 0, 0, .1, 0, 0, 0, .1;
	Mat sigmaV(m, m);
	sigmaV << .1;
	Vec initState(n);
	initState << 1, 0, 0;
	Mat initVar(n, n);
	initVar << .1, 0, 0, 0, .1, 0, 0, 0, .1;

	//data generation
	std::string filename = ".\\LDS1\\tree2_";
	int i = 0;
	std::ofstream outTree(filename + std::to_string(i) + ".txt");
	std::ofstream trueState(filename + std::to_string(i) + "_true.txt");
	int current = 0;
	generateOneLDSTree(n, m, current, A, C, sigmaW, sigmaV, generation, -1, initState, outTree, trueState, [] {return 2; });

	//initial condition

	//Mat a(n, n);
	//a << 0.501276, -0.915556, 0.8482, 0.498875;
	//Mat c(m, n);
	//c << -0.00300284, -0.173368, 0.169364, -0.00213864;
	//Mat sigmaw(n,n);
	//sigmaw << 0.00891825, -0.00469345, -0.00469345, 0.01412;
	//Mat sigmav(m,m);
	//sigmav << 0.000513069, 0.000209337, 0.000209337, 0.0003881866;
	//Vec initMean(n);
	//initMean << -0.468277, -5.23107;
	

	//load
	TreeLDS t;
	t.load(".\\LDS1\\tree2_0.txt", n, m);
	//t.load(".\\LDS1\\tree2_ex2.txt", n, m);

	//estimation
	std::ofstream out(".\\LDS1\\tree2Res.txt");
	estimateLDS(t, A, C, sigmaW, sigmaV, initState, initVar, out);
}

void carpetBombingTest() {
	int generation = 10;
	int seekNo = 100;
	double parameterMin = -2.0;
	double parameterMax = 2.0;
	double covarMin = 0.1;
	double covarMax = 1.0;

	//int n = 2;
	//int m = 2;
	//Mat A(n, n);
	//A << 1, 0, 0, 1;
	//Mat C(m, n);
	//C << 1, 0, 0, 1;
	//Mat sigmaW(n, n);
	//sigmaW << .5, 0, 0, .5;
	//Mat sigmaV(m, m);
	//sigmaV << .5, 0, 0, .5;
	//Vec initState(2);
	//initState << 1, 1;

	//tree2

	int n = 2;
	int m = 2;
	double theta = 3.1415 / 3.0;
	Mat A(n, n);
	A << cos(theta), -sin(theta), sin(theta), cos(theta);
	Mat C(m, n);
	C << 1, 0, 0, 1;
	Mat sigmaW(n, n);
	sigmaW << .01, 0, 0, .01;
	Mat sigmaV(m, m);
	sigmaV << .01, 0, 0, .01;
	Vec initState(n);
	initState << 1, 0;
	Mat initVar(n, n);
	initVar << 1, 0, 0, 1;

	////tree 3
	//int n = 2;
	//int m = 1;
	//double theta = 3.1415 / 3.0;
	//Mat A(n, n);
	//A << 1, 1, 0, 1;
	//Mat C(m, n);
	//C << 1, 0;
	//Mat sigmaW(n, n);
	//sigmaW << .1, 0, 0, .1;
	//Mat sigmaV(m, m);
	//sigmaV << .1;
	//Vec initState(2);
	//initState << 1, 10;

	////tree5
	//int n = 1;
	//int m = 1;
	//Mat A(n, n);
	//A << 1;
	//Mat C(m, n);
	//C << 1;
	//Mat sigmaW(n, n);
	//sigmaW << .1;
	//Mat sigmaV(m, m);
	//sigmaV << .1;
	//Vec initState(2);
	//initState << 1;

	//tree 6
	//int n = 3;
	//int m = 1;
	//Mat A(n, n);
	//A << 1, 0, 0, 0, 1, 0, 0, 0, 1;
	//Mat C(m, n);
	//C << 1, 0, 0;
	//Mat sigmaW(n, n);
	//sigmaW << .1, 0, 0, 0, .1, 0, 0, 0, .1;
	//Mat sigmaV(m, m);
	//sigmaV << .1;
	//Vec initState(2);
	//initState << 1, 0, 0;
	//Mat initVar(n, n);
	//initVar << .1, 0, 0, 0, .1, 0, 0, 0, .1;

	//data generation
	std::string filename = ".\\LDS1\\tree2_";
	int i = 0;
	std::ofstream outTree(filename + std::to_string(i) + ".txt");
	std::ofstream trueState(filename + std::to_string(i) + "_true.txt");
	int current = 0;
	generateOneLDSTree(n, m, current, A, C, sigmaW, sigmaV, generation, -1, initState, outTree, trueState, [] {return 2; });

	//load
	TreeLDS t;
	t.load(".\\LDS1\\tree2_0.txt", n, m);

	//carpetBombing
	double logLik = carpetBombing(t, seekNo, parameterMin, parameterMax, covarMin, covarMax, A, C, sigmaW, sigmaV, initState, initVar);

	//output
	std::ofstream outFile(".\\LDS1\\carpet_res.txt");
	std::ofstream out(".\\LDS1\\carpet_smoothe.txt");

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
	
	t.smoothing(A, C, sigmaW, sigmaV, initState, initVar);

	t.output(out);
	std::cout << t.logLik() << std::endl;
}

void carpetBombingForestTest() {
	int generation = 10;
	int seekNo = 20;
	double parameterMin = -2.0;
	double parameterMax = 2.0;
	double covarMin = 0.1;
	double covarMax = 1.0;

	//int n = 2;
	//int m = 2;
	//Mat A(n, n);
	//A << 1, 0, 0, 1;
	//Mat C(m, n);
	//C << 1, 0, 0, 1;
	//Mat sigmaW(n, n);
	//sigmaW << .5, 0, 0, .5;
	//Mat sigmaV(m, m);
	//sigmaV << .5, 0, 0, .5;
	//Vec initState(2);
	//initState << 1, 1;

	//tree2

	int n = 2;
	int m = 2;
	double theta = 3.1415 / 3.0;
	Mat A(n, n);
	A << cos(theta), -sin(theta), sin(theta), cos(theta);
	Mat C(m, n);
	C << 1, 0, 0, 1;
	Mat sigmaW(n, n);
	sigmaW << .1, 0, 0, .1;
	Mat sigmaV(m, m);
	sigmaV << .5, 0, 0, .5;
	Vec initState(2);
	initState << 1, 0;

	//tree 3
	//int n = 2;
	//int m = 1;
	//double theta = 3.1415 / 3.0;
	//Mat A(n, n);
	//A << 1, 1, 0, 1;
	//Mat C(m, n);
	//C << 1, 0;
	//Mat sigmaW(n, n);
	//sigmaW << .1, 0, 0, .1;
	//Mat sigmaV(m, m);
	//sigmaV << .1;
	//Vec initState(2);
	//initState << 1, 10;

	////tree5
	//int n = 1;
	//int m = 1;
	//Mat A(n, n);
	//A << 1;
	//Mat C(m, n);
	//C << 1;
	//Mat sigmaW(n, n);
	//sigmaW << .1;
	//Mat sigmaV(m, m);
	//sigmaV << .1;
	//Vec initState(2);
	//initState << 1;

	//data generation
	std::string filename = ".\\LDS1\\tree2_";
	for (int i = 0; i != 5; i++) {
		std::ofstream outTree(filename + std::to_string(i) + ".txt");
		std::ofstream trueState(filename + std::to_string(i) + "_true.txt");
		int current = 0;
		generateOneLDSTree(n, m, current, A, C, sigmaW, sigmaV, generation, -1, initState, outTree, trueState, [] {return 2; });
	}

	//initial condition
	Mat initVar(n, n);
	initVar << .1, 0, 0, .1;

	//load
	ForestLDS t(n,m);
	for (int i = 0; i != 5; i++) {
		TreeLDS* tree = new TreeLDS;
		tree->load(filename + std::to_string(i) + ".txt", n, m);
		t.add(tree);
	}

	//carpetBombing
	double logLik = carpetBombing(t, seekNo, parameterMin, parameterMax, covarMin, covarMax, A, C, sigmaW, sigmaV, initState, initVar);

	//output
	std::ofstream outFile(".\\LDS1\\carpet_res.txt");
	std::ofstream out(".\\LDS1\\carpet_smoothe.txt");

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

	t.smoothing(A, C, sigmaW, sigmaV, initState, initVar);

	t.output(out);
	std::cout << t.logLik() << std::endl;
}



void pushWakamotoTreesLDS(ForestLDS& forest, std::string namebase, int treeNo) {  //Be aware, this function allocates memory for trees in forest
	int n = forest.n;
	int m = 1;

	for (int i = 0; i != treeNo; i++) {
		TreeLDS* t = new TreeLDS();
		std::string filename = namebase + std::to_string(i) + ".txt";
		t->load_w(filename, n);
		if (t->size() > 10)
			forest.add(t);
	}
}

void ReadWakamotoLDS(std::string experiment, ForestLDS& forest) {
	if (experiment == "F3_rpsL_Glc_30C_3min") {
		pushWakamotoTreesLDS(forest, ".\\RawData\\F3_rpsL_Glc_30C_3min\\Results0013.xls_w_size_gfp", 22);
		pushWakamotoTreesLDS(forest, ".\\RawData\\F3_rpsL_Glc_30C_3min\\Results0072.xls_w_size_gfp", 24);
		pushWakamotoTreesLDS(forest, ".\\RawData\\F3_rpsL_Glc_30C_3min\\Results0253.xls_w_size_gfp", 35);
		pushWakamotoTreesLDS(forest, ".\\RawData\\F3_rpsL_Glc_30C_3min\\Results0284.xls_w_size_gfp", 23);
	}
	else if(experiment == "F3_rpsL_Glc_37C_1min"){
		//pushWakamotoTreesLDS(forest, ".\\RawData\\F3_rpsL_Glc_37C_1min\\Results0008.xls", 20);
		pushWakamotoTreesLDS(forest, ".\\RawData\\F3_rpsL_Glc_37C_1min\\Results0012.xls_w_size_gfp", 28);
	}
	else if (experiment == "Br_Gly_37C_1min") {
		pushWakamotoTreesLDS(forest, ".\\RawData\\Br_Gly_37C_1min\\Results0003.xls_w_size_gfp", 16);
	}
	else if (experiment == "F3_LVS4_Glc_37C_1min") {
		pushWakamotoTreesLDS(forest, ".\\RawData\\F3_LVS4_Glc_37C_1min\\Results0002.xls_w_size_gfp", 25);
	}
	else if (experiment == "Br_Glc_37C_1min") {
		pushWakamotoTreesLDS(forest, ".\\RawData\\Br_Glc_37C_1min\\Results0020.xls", 15);
	}
	else if (experiment == "F3_T7Venus_Glc30C_3min") {
		pushWakamotoTreesLDS(forest, ".\\RawData\\F3_T7Venus_Glc30C_3min\\Results0001.xls_w_size_gfp", 27);
		pushWakamotoTreesLDS(forest, ".\\RawData\\F3_T7Venus_Glc30C_3min\\Results0004.xls_w_size_gfp", 14);
		pushWakamotoTreesLDS(forest, ".\\RawData\\F3_T7Venus_Glc30C_3min\\Results0005.xls_w_size_gfp", 12);
	}
	else {
		std::cout << "Such an experiment does NOT exist..." << std::endl;
		abort();
	}
}


void WakamotoEstimatorLDS(std::string experiment, std::string resultFile, int n) {
	//int generation = 10;
	int seekNo = std::min(3000, std::max(20, (int)pow(5, n)));
	//int seekNo = 500;
	//if (n > 2)
	//	seekNo = 5000;
	double parameterMin = -5.0;
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
	ReadWakamotoLDS(experiment, forest);

	//carpetBombing
	double logLik = carpetBombing(forest, seekNo, parameterMin, parameterMax, covarMin, covarMax, A, C, sigmaW, sigmaV, initState, initVar, true);

	//output
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

	forest.smoothing(A, C, sigmaW, sigmaV, initState, initVar);

	forest.output();
}

void WakamotoEstimatorLDSwithMeanV(std::string experiment, std::string resultFile, int n) {
	//int generation = 10;
	//int seekNo = std::min(3000, std::max(20, (int)pow(10, n)));
	int seekNo = 1000;
	if (n == 1 || n == 2)
		seekNo = 30;
	//int seekNo = 500;
	//if (n > 2)
	//	seekNo = 5000;
	double parameterMin = -5.0;
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
	Vec meanV(m);


	//Read
	ForestLDS forest(n, m);
	ReadWakamotoLDS(experiment, forest);

	//carpetBombing
	double logLik = carpetBombing(forest, seekNo, parameterMin, parameterMax, covarMin, covarMax, A, C, sigmaW, meanV, sigmaV, initState, initVar, true);

	//output
	std::ofstream outFile(resultFile);

	outFile << logLik << " ";
	outFile << "A: ";
	matOut(A, outFile);
	outFile << "C: ";
	matOut(C, outFile);
	outFile << "sigmaW: ";
	matOut(sigmaW, outFile);
	outFile << "meanV: ";
	vecOut(meanV, outFile);
	outFile << "sigmaV: ";
	matOut(sigmaV, outFile);
	outFile << "rootMean: ";
	vecOut(initState, outFile);
	outFile << "rootVar: ";
	matOut(initVar, outFile);
	outFile << std::endl;

	forest.smoothing(A, C, sigmaW, meanV, sigmaV, initState, initVar);

	forest.output();
}

void MatRead(int m, int n, Mat& mat, std::istream& in) {
	std::string temp;
	in >> temp; //trash name

	for (int i = 0; i != m; i++) {
		for (int j = 0; j != n; j++) {
			in >> temp;
			temp.erase(temp.size() - 1);
			mat(i, j) = std::stod(temp);
		}
	}
}

void VecRead(int n, Vec& mat, std::istream& in) {
	std::string temp;
	in >> temp; //trash name

	for (int j = 0; j != n; j++) {
		in >> temp;
		temp.erase(temp.size() - 1);
		mat[j] = std::stod(temp);
	}

}

void loadEstimatedParameters(std::string filename, int n, int m, Mat& A, Mat& C, Mat& sigmaW, Mat& sigmaV, Vec& initState, Mat& initVar) {
	
	//initializations
	A = sigmaW = initVar = Mat::Zero(n, n);
	C = Mat::Zero(m, n);
	sigmaV = Mat::Zero(m, m);
	initState = Vec::Zero(n);

	std::ifstream in(filename);
	std::string temp;
	in >> temp; //trash log-like
	MatRead(n, n, A, in);
	MatRead(m, n, C, in);
	MatRead(n, n, sigmaW, in);
	MatRead(m, m, sigmaV, in);
	VecRead(n, initState, in);
	MatRead(n, n, initVar, in);
}

void loadEstimatedParameters(std::string filename, int n, int m, Mat& A, Mat& C, Mat& sigmaW, Vec& meanV, Mat& sigmaV, Vec& initState, Mat& initVar) {

	//initializations
	A = sigmaW = initVar = Mat::Zero(n, n);
	C = Mat::Zero(m, n);
	sigmaV = Mat::Zero(m, m);
	initState = Vec::Zero(n);
	meanV = Vec::Zero(m);

	std::ifstream in(filename);
	std::string temp;
	in >> temp; //trash log-like
	MatRead(n, n, A, in);
	MatRead(m, n, C, in);
	MatRead(n, n, sigmaW, in);
	VecRead(m, meanV, in);
	MatRead(m, m, sigmaV, in);
	VecRead(n, initState, in);
	MatRead(n, n, initVar, in);
}


void CheckEstimatedResult(std::string experiment) {
	//estimated parameters
	int n = 3;
	int m = 1;
	Mat A;
	Mat C;
	Mat sigmaW;
	Vec meanV;  //add on 24/5/2018
	Mat sigmaV;
	Vec initState;
	Mat initVar;
	loadEstimatedParameters(".\\LDS1\\wakamotoRes\\" + experiment + "\\" + experiment + "_normalize_for_cross_valid_symmetrized_nonRoot_n="  +  std::to_string(n) + ".txt",
		n, m, A, C, sigmaW, sigmaV, initState, initVar);
	//loadEstimatedParameters(".\\LDS1\\wakamotoRes\\" + experiment + "\\" + experiment + "_fixedMeanV_nonDiagonalizationOfSigmaW_n=" + std::to_string(n) + ".txt",
	//	n, m, A, C, sigmaW, meanV, sigmaV, initState, initVar);


	//Read
	ForestLDS forest(n, m);
	ReadWakamotoLDS(experiment, forest);

	forest.smoothing(A, C, sigmaW, sigmaV, initState, initVar);
	//forest.smoothing(A, C, sigmaW, meanV, sigmaV, initState, initVar);

	std::cout << forest.logLik() << std::endl;

	forest.checkEstimatedResult();

	//forest.execSummation(C, sigmaV);
	//double numer = forest.tptc - (C * forest.zk)(0, 0) * (C * forest.zk)(0, 0);
	//double denom = forest.tktk - (C * forest.zk)(0, 0) * (C * forest.zk)(0, 0);
	//std::cout << (exp(numer * denom) - 1) / (exp(denom) - 1)  << std::endl;

	//std::ofstream paper(".\\paper\\res\\" + experiment + "_meanV.txt");
	//forest.outputForPaper(paper);

	//output of tree 
	std::string filenameBase = ".\\paper\\res\\" + experiment + "n=" + std::to_string(n) + "_";
	forest.outputForPaper(filenameBase);
}

void CheckSmoothedPath(std::string experiment) {
	int itr = 100; //how many times sample the path 

	//estimated parameters
	int n = 3;
	int m = 1;
	Mat A;
	Mat C;
	Mat sigmaW;
	Mat sigmaV;
	Vec initState;
	Mat initVar;
	loadEstimatedParameters(".\\LDS1\\wakamotoRes\\" + experiment + "\\" + experiment + "_normalize_for_cross_valid_symmetrized_nonRoot_n=" + std::to_string(n) + ".txt",
		n, m, A, C, sigmaW, sigmaV, initState, initVar);


	//Read
	ForestLDS forest(n, m);
	ReadWakamotoLDS(experiment, forest);

	forest.smoothing(A, C, sigmaW, sigmaV, initState, initVar);

	std::ofstream out("Plan3_tau" + experiment + ".txt");
	for (int i = 0; i != itr; i++) {
		forest.samplePath(C, sigmaV);
		forest.outputSamplePath(out);
	}
}


void checkDistributions() { //check estimated distribution and correlated functions;

	//consts
	int sequenceLength = 100;
	int sequenceNo = 1000;

	//estimated parameters
	int n = 3;
	int m = 1;
	Mat A;
	Mat C;
	Mat sigmaW;
	Mat sigmaV;
	Vec initState;
	Mat initVar;
	loadEstimatedParameters(".\\LDS1\\wakamotoRes\\F3_LVS4_Glc_37C_1min\\F3_LVS4_Glc_37C_1min_normalize_for_cross_valid_symmetrized_nonRoot_n=" + std::to_string(n) + ".txt",
		n, m, A, C, sigmaW, sigmaV, initState, initVar);
	//output file
	std::ofstream outFile(".\\LDS1\\checkDistributions.txt");

	//generate forward sequence and record
	for (int i = 0; i != sequenceNo; i++) {
		//initial state
		//Vec current = normal_random_variable(initState, initVar)();
		Vec current = initState;
		double observed = normal_random_variable(C * current, sigmaV)()[0];

		for (int j = 0; j != sequenceLength; j++) {
			//next state
			Vec next = normal_random_variable(A * current, sigmaW)();
			double observedNext = normal_random_variable(C * next, sigmaV)()[0];

			//output
			outFile << observed << " " << exp(observed) << " " << observedNext << " " << exp(observedNext) << std::endl;

			//update current state
			current = next;
			observed = observedNext;
		}
	}
}

void checkDistributionsSteady(std::string experiment) { //check estimated distribution and correlated functions;
	
	//consts
	int expNo = 1000000;

	//estimated parameters
	int n = 3;
	int m = 1;
	Mat A;
	Mat C;
	Mat sigmaW;
	Mat sigmaV;
	Vec initState;
	Mat initVar;
	loadEstimatedParameters(".\\LDS1\\wakamotoRes\\" + experiment + "\\" + experiment + "_normalize_for_cross_valid_symmetrized_nonRoot_n=" + std::to_string(n) + ".txt",
		n, m, A, C, sigmaW, sigmaV, initState, initVar);

	//output file
	std::ofstream outFile(".\\LDS1\\checkDistributionsSteady.txt");
	
	//estimate steady distribution from data
	ForestLDS forest(n, m);
	ReadWakamotoLDS(experiment, forest);

	forest.smoothing(A, C, sigmaW, sigmaV, initState, initVar);

	Vec zk = Eigen::VectorXd::Zero(n);
	Mat zkzkt = Eigen::MatrixXd::Zero(n, n);
	
	double sum = 0.0;
	for (auto t : forest.forest) {
		double frac = t->size() / (double) forest.nodeSize();

		t->execSummation();
		zk += t->zk * frac;
		zkzkt += t->zkzkt * frac;
	}

	Mat covar = zkzkt - zk * zk.transpose();

	//generate forward sequence and record
	for (int i = 0; i != expNo; i++) {
		Vec current = normal_random_variable(zk, covar)();
		Vec next = normal_random_variable(A * current, sigmaW)();
		double observed = normal_random_variable(C * current, sigmaV)()[0];
		double observedNext = normal_random_variable(C * next, sigmaV)()[0];
	
		//output
		outFile << observed << " " << exp(observed) << " " << observedNext << " " << exp(observedNext) << std::endl;
	}
}

void matrixTest() {
	Eigen::MatrixXd A(3, 3);
	A << 1, .1, .2, .1, 2, .3, .2, .3, 3;
	
	Eigen::JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
	std::cout << svd.singularValues() << std::endl;
	std::cout << svd.matrixU() << std::endl;
	std::cout << svd.matrixV() << std::endl;
	std::cout << svd.matrixU() * svd.matrixU().transpose() << std::endl;
	//std::cout << svd.singularValues().asDiagonal() << std::endl;
	std::cout << svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixU().transpose() << std::endl;
}

void generateLDSForest() {
	int generation = 10;
	int seekNo = 20;

	//int n = 2;
	//int m = 2;
	//Mat A(n, n);
	//A << 1, 0, 0, 1;
	//Mat C(m, n);
	//C << 1, 0, 0, 1;
	//Mat sigmaW(n, n);
	//sigmaW << .5, 0, 0, .5;
	//Mat sigmaV(m, m);
	//sigmaV << .5, 0, 0, .5;
	//Vec initState(2);
	//initState << 1, 1;

	//tree2

	//int n = 2;
	//int m = 2;
	//double theta = 3.1415 / 3.0;
	//Mat A(n, n);
	//A << cos(theta), -sin(theta), sin(theta), cos(theta);
	//Mat C(m, n);
	//C << 1, 0, 0, 1;
	//Mat sigmaW(n, n);
	//sigmaW << .1, 0, 0, .1;
	//Mat sigmaV(m, m);
	//sigmaV << .5, 0, 0, .5;
	//Vec initState(2);
	//initState << 1, 0;

	//tree 3
	//int n = 2;
	//int m = 1;
	//double theta = 3.1415 / 3.0;
	//Mat A(n, n);
	//A << 1, 1, 0, 1;
	//Mat C(m, n);
	//C << 1, 0;
	//Mat sigmaW(n, n);
	//sigmaW << .1, 0, 0, .1;
	//Mat sigmaV(m, m);
	//sigmaV << .1;
	//Vec initState(2);
	//initState << 1, 10;

	//tree5
	int n = 1;
	int m = 1;
	Mat A(n, n);
	A << 0;
	Mat C(m, n);
	C << 0;
	Mat sigmaW(n, n);
	sigmaW << 0;
	Mat sigmaV(m, m);
	sigmaV << 1;
	Vec initState(n);
	initState << 0;

	//data generation
	std::string filename = ".\\paper\\tree\\LDSiid\\LDSiidWithoutMean";
	for (int i = 0; i != 10; i++) {
		std::ofstream outTree(filename + std::to_string(i) + ".txt");
		std::ofstream trueState(filename + std::to_string(i) + "_true.txt");
		int current = 0;
		generateOneLDSTree(n, m, current, A, C, sigmaW, sigmaV, generation, -1, initState, outTree, trueState, [] {return 2; });
	}

}

void carpetBombingForestTestForGeneratedData(int n, int m) {
	int generation = 10;
	int seekNo = 20;
	if (n > 2)
		seekNo = 20;
	double parameterMin = -2.0;
	double parameterMax = 2.0;
	double covarMin = 0.1;
	double covarMax = 1.0;


	Mat A(n, n);
	Mat C(m, n);
	Mat sigmaW(n, n);
	Mat sigmaV(m, m);
	Vec initState(n);

	//initial condition
	Mat initVar(n, n);

	//load
	ForestLDS t(n, m);
	std::string filename = ".\\paper\\tree\\LDSiid\\LDSiid";
	for (int i = 0; i != 1; i++) {
		TreeLDS* tree = new TreeLDS;
		tree->load(filename + std::to_string(i) + ".txt", n, m);
		t.add(tree);
	}

	//carpetBombing
	double logLik = carpetBombing(t, seekNo, parameterMin, parameterMax, covarMin, covarMax, A, C, sigmaW, sigmaV, initState, initVar);

	//output
	std::ofstream outFile(".\\paper\\carpet_res_" + std::to_string(n) + ".txt");
	std::ofstream out(".\\paper\\carpet_smoothe_" + std::to_string(n) + ".txt");

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

	t.smoothing(A, C, sigmaW, sigmaV, initState, initVar);

	t.output(out);
	std::cout << t.logLik() << std::endl;

	t.checkEstimatedResult();

	//forest.execSummation(C, sigmaV);
	//double numer = forest.tptc - (C * forest.zk)(0, 0) * (C * forest.zk)(0, 0);
	//double denom = forest.tktk - (C * forest.zk)(0, 0) * (C * forest.zk)(0, 0);
	//std::cout << (exp(numer * denom) - 1) / (exp(denom) - 1)  << std::endl;

	//std::ofstream paper(".\\paper\\res\\" + experiment + "_meanV.txt");
	//forest.outputForPaper(paper);

	//output of tree 
	std::string filenameBase = ".\\paper\\LDSiid_n=" + std::to_string(n) + "_";
	t.outputForPaper(filenameBase);
}

double carpetBombingGegeratedDateReportingAIC(int n, int m) {
	int generation = 15;
	int seekNo = 10;
	if (n > 2)
		seekNo = 10;
	double parameterMin = -2.0;
	double parameterMax = 2.0;
	double covarMin = 0.1;
	double covarMax = 1.0;


	Mat A(n, n);
	Mat C(m, n);
	Mat sigmaW(n, n);
	Mat sigmaV(m, m);
	Vec initState(n);

	//initial condition
	Mat initVar(n, n);

	//load
	ForestLDS t(n, m);
	std::string filename = ".\\paper\\tree\\LDSiid\\LDSiid";
	for (int i = 0; i != 1; i++) {
		TreeLDS* tree = new TreeLDS;
		tree->load(filename + std::to_string(i) + ".txt", n, m);
		t.add(tree);
	}

	//carpetBombing
	double logLik = carpetBombing(t, seekNo, parameterMin, parameterMax, covarMin, covarMax, A, C, sigmaW, sigmaV, initState, initVar);

	return logLik - 2 * n * n - 2 * n - 1;  //equivalentform of AIC
}

double log_likelihood_1dim(double x, double mean, double var) {
	return -0.5 * log(2.0 * 3.1415926 * var) - (x - mean) * (x - mean) / 2.0 / var;
}

double MLElikelihood(const std::vector<double>& v) {
	double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();

	double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
	double var = sq_sum / v.size() - mean * mean;


	double res = 0.0;
	for (double d : v)
		res += log_likelihood_1dim(d, mean, var);

	return res;
}

int AICselectionGeneratedData(int maxDim, std::ofstream& out) {
	generateLDSForest();

	//i = 0 (MLE)
	int n = 1; int m = 1;
	ForestLDS t(n, m);
	std::string filename = ".\\paper\\tree\\LDSiid\\LDSiid";
	for (int i = 0; i != 1; i++) {
		TreeLDS* tree = new TreeLDS;
		tree->load(filename + std::to_string(i) + ".txt", n, m);
		t.add(tree);
	}

	std::vector<double> gentimes = t.forest[0]->gentimes();
	//std::vector<double> logGentimes;
	//for (double g : gentimes)
	//	logGentimes.push_back(log(g));

	int minDim = 0;
	double minAIC = MLElikelihood(gentimes) - 2;  //it should have defined as "maxAIC"......

	out << minAIC << std::endl;

	for (int i = 1; i != maxDim; i++) {
		double AIC = carpetBombingGegeratedDateReportingAIC(i, 1);
		if (minAIC < AIC) {
			minDim = i;
			minAIC = AIC;
		}

		out << AIC << std::endl;
	}
	out << std::endl;
	return minDim;
}

void AICselctions(int iteration, int maxDim) {
	std::string filenameBase = ".\\paper\\res\\AIC";
	std::ofstream histo(".\\paper\\res\\histo.txt");

	for (int i = 0; i != iteration; i++) {
		std::ofstream out(filenameBase + std::to_string(i) + ".txt");
		int minDim = AICselectionGeneratedData(maxDim, out);
		histo << minDim << " ";
	}
	histo << std::endl;
}

//void gentimeSwap(ForestLDS& f) { //randomly swap generationtimes of trees
//	assert(f.m == 1);
//
//	//collect all generation times
//	std::vector<double> gentimes;
//	for (TreeLDS* t : f.forest) {
//		std::vector<double> thisTreeGentimes = t->gentimes();
//		gentimes.insert(gentimes.end(), thisTreeGentimes.begin(), thisTreeGentimes.end());
//	}
//
//	//swapping
//	for (TreeLDS * t : f.forest) 
//		t->swapGentime(gentimes);
//}


void bootstrap_estimate_core(std::string experiment, std::string resultFile, int n, const std::vector<double> gentimes, std::vector<std::vector<int>>& itrs) {
	//int generation = 10;
	int seekNo = std::min(3000, std::max(20, (int)pow(5, n)));
	//int seekNo = 500;
	//if (n > 2)
	//	seekNo = 5000;
	double parameterMin = -5.0;
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
	ReadWakamotoLDS(experiment, forest);

	//generation time swapping
	for (int i = 0; i != forest.forest.size(); i++)
		(forest.forest[i])->swapGentime(gentimes, itrs[i]);

	//carpetBombing
	double logLik = carpetBombing(forest, seekNo, parameterMin, parameterMax, covarMin, covarMax, A, C, sigmaW, sigmaV, initState, initVar, true);

	//output
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

	forest.smoothing(A, C, sigmaW, sigmaV, initState, initVar);

	forest.output();
}

double bootstrap_estimate_core_reporting_AIC(std::string experiment, int n, const std::vector<double> gentimes, std::vector<std::vector<int>>& itrs) {
	//int generation = 10;
	int seekNo = std::min(3000, std::max(20, (int)pow(5, n)));
	//int seekNo = 500;
	//if (n > 2)
	//	seekNo = 5000;
	double parameterMin = -5.0;
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
	ReadWakamotoLDS(experiment, forest);

	//generation time swapping
	for (int i = 0; i != forest.forest.size(); i++)
		(forest.forest[i])->swapGentime(gentimes, itrs[i]);

	//carpetBombing
	double logLik = carpetBombing(forest, seekNo, parameterMin, parameterMax, covarMin, covarMax, A, C, sigmaW, sigmaV, initState, initVar, true);

	//output
	//std::ofstream outFile(resultFile);

	//outFile << logLik << " ";
	//outFile << "A: ";
	//matOut(A, outFile);
	//outFile << "C: ";
	//matOut(C, outFile);
	//outFile << "sigmaW: ";
	//matOut(sigmaW, outFile);
	//outFile << "sigmaV: ";
	//matOut(sigmaV, outFile);
	//outFile << "rootMean: ";
	//vecOut(initState, outFile);
	//outFile << "rootVar: ";
	//matOut(initVar, outFile);
	//outFile << std::endl;

	//forest.smoothing(A, C, sigmaW, sigmaV, initState, initVar);

	//forest.output();

	return logLik - 2 * n * n - 2 * n - 1;
}

void WakamotoEstimatorLDS_Bootstrap(std::string experiment, std::string resultFile) {

	//Read
	ForestLDS f(1, 1);
	ReadWakamotoLDS(experiment, f);

	//collect all generation times
	std::vector<double> gentimes;
	for (TreeLDS* t : f.forest) {
		std::vector<double> thisTreeGentimes = t->gentimes();
		gentimes.insert(gentimes.end(), thisTreeGentimes.begin(), thisTreeGentimes.end());
	}

	//make random iterator sequences
	std::random_device rnd;
	std::mt19937_64 randGen(rnd());
	std::uniform_int_distribution<int> uniform(0, gentimes.size() - 1);

	std::vector<std::vector<int>> itrs;
	for (int i = 0; i != f.forest.size(); i++) {
		std::vector<int> itr;
		for (int j = 0; j != f.forest[i]->size(); j++) {
			itr.push_back(uniform(randGen));
		}
		itrs.push_back(itr);
	}

	//execute estimation
	for (int i = 1; i != 11; i++) {
		std::string newFileName = resultFile + std::to_string(i) + ".txt";
		bootstrap_estimate_core(experiment, newFileName, i, gentimes, itrs);
	}
}

int AICselectionBootStrap(int maxDim, std::string experiment, std::ofstream& out) {
	generateLDSForest();

	int n = 1; int m = 1;
	ForestLDS forest(n, m);
	ReadWakamotoLDS(experiment, forest);

	std::vector<double> gentimes;
	for (TreeLDS* t : forest.forest) {
		std::vector<double> thisTreeGentimes = t->gentimes();
		gentimes.insert(gentimes.end(), thisTreeGentimes.begin(), thisTreeGentimes.end());
	}
	std::vector<double>& logGentimes = gentimes;  //already log is taken
	//for (double g : gentimes)
	//	logGentimes.push_back(log(g));



	//swapping and other dimensions
	//make random iterator sequences
	std::random_device rnd;
	std::mt19937_64 randGen(rnd());
	std::uniform_int_distribution<int> uniform(0, gentimes.size() - 1);

	std::vector<double> swappedGentimes; //for i = 0

	std::vector<std::vector<int>> itrs;
	for (int i = 0; i != forest.forest.size(); i++) {
		std::vector<int> itr;
		for (int j = 0; j != forest.forest[i]->size(); j++) {
			double gentime = uniform(randGen);
			itr.push_back(gentime);
			swappedGentimes.push_back(gentimes[gentime]);

		}
		itrs.push_back(itr);
	}

	//i = 0 (MLE)
	int minDim = 0;
	double minAIC = MLElikelihood(swappedGentimes) - 2;  //it should have defined as "maxAIC"......

	out << minAIC << std::endl;

	//execute estimation
	for (int i = 1; i != maxDim + 1; i++) {
		//std::string newFileName = resultFile + std::to_string(i) + ".txt";
		double AIC = bootstrap_estimate_core_reporting_AIC(experiment, i, gentimes, itrs);
		if (minAIC < AIC) {
			minDim = i;
			minAIC = AIC;
		}

		out << AIC << std::endl;
	}

	out << std::endl;

	return minDim;
}

void AICselctionsBootStrap(int iteration, std::string experiment, int maxDim) {
	std::string filenameBase = ".\\paper\\res\\AIC_bootstrap";
	std::ofstream histo(".\\paper\\res\\histoBootStrap.txt");

	for (int i = 0; i != iteration; i++) {
		std::ofstream out(filenameBase + std::to_string(i) + ".txt");
		int minDim = AICselectionBootStrap(maxDim, experiment, out);
		histo << minDim << " ";
	}
	histo << std::endl;
}