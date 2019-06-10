#pragma once

#include <Eigen/Core>
#include <functional>
#include <random>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

void generateTrees();
void estimateTree1();
void estimateTree2();
void generateTree2forest();
void estimateTree3();
void generateTree3forest();
//void generateLDSForest();
void generateOneLDSTree(
	int n,
	int m,
	int & current, // # of written cells
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::MatrixXd& sigmaV,
	int generation,
	int parent,
	const Eigen::VectorXd& currentState,
	std::ofstream& file,
	std::ofstream& trueState,
	std::function<int()> childrenNo = [] {return 2; }
);

//copied (and modified) from stackoverflow https://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c

class normal_random_variable
{
public:
	normal_random_variable(Eigen::MatrixXd const& covar)
		: normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
	{}

	normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
		: mean(mean)
	{
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
		transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
	}
	Eigen::VectorXd operator()() const
	{
		static std::mt19937 gen{ std::random_device{}() };
		static std::normal_distribution<> dist;

		return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
	}

private:
	Eigen::VectorXd mean;
	Eigen::MatrixXd transform;
};