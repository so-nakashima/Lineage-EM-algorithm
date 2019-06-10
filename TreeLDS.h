#pragma once

#include<vector>
#include<string>
#include "Density.h"
#include <functional>
#include <Eigen/Core>
#include <iostream>
#include <fstream>

//set true if you want to exclude the root node from analysis
const bool excludeRoot = true;

class TreeLDS;

class NodeLDS
{
public:
	//constructors
	NodeLDS(); //not for use
	NodeLDS::NodeLDS(Eigen::VectorXd observed, int parentsID, bool isLeaf, int N, int M, std::vector<NodeLDS>* tree); //for internal nodes
	NodeLDS::NodeLDS(Eigen::VectorXd observed, int parentsID, bool isLeaf, int N, int M, double firstSize, double lastSize, double firstGFP, double lastGFP, double meanGFP, std::vector<NodeLDS>* tree); //for internal nodes,
	//NodeLDS(int type, double survivalTime, int typeno, int parentsId, std::vector<NodeLDS>* tree); //for leaves  //ignored currently
	~NodeLDS();

	//basic tree operations
	std::vector<int> children; //return children's IDs
	int parent() const { return o_parent; } //return parent's ID
	//int sister() const;  //return sister's ID
	Eigen::VectorXd observed; //return X in my note  //FOR LINEAGE DATA, the first component must be the generation time.
	//double bornTime() { return o_borntime; }
	bool& isLeaf() { return o_isLeaf; }
	void addChild(int childsID); //return whether successfully added the child. Left first, Right last.
	//bool isleft() const; // return whether or not this is left child (root is false)
	bool isroot() const { return this == &o_tree->at(0); } //return whether this is root;

	//dimensions
	int n; //dimension of hidden variable
	int m; //dimension of observed variable

	//inference
	void updateFoward(
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::MatrixXd& sigmaV
	);
	bool updateFilter(
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::MatrixXd& sigmaV
	); 
	bool updateFilter(
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::VectorXd& meanV,
		const Eigen::MatrixXd& sigmaV
	);
	void updateSmoothe(
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::MatrixXd& sigmaV
	);
	Eigen::VectorXd smoothe_mean;  //smoothed distribution
	Eigen::MatrixXd smoothe_var;
	Eigen::VectorXd filter_mean;  //filter distribution
	Eigen::MatrixXd filter_var;
	Eigen::VectorXd forward_mean;  //unconditioned distribution of hidden variable
	Eigen::MatrixXd forward_var;
	Eigen::MatrixXd J; //used in M step
	double log_c() const { return o_logc; } //return c; used to calculate log-likelihood
	//void changeC(double c) { o_c = c; }

	//output
	void output(std::ofstream& file);
	//debug
	Eigen::VectorXd predObserved; //C * smoothe_mean
	Eigen::VectorXd Cfilter;
	Eigen::VectorXd forcasted;
	Eigen::VectorXd sz; //sample path of zk from smoothing distribution
	double st; //sampled tau from smoothing ditribution
	void extendSmoothePath(const Eigen::MatrixXd& C, const Eigen::MatrixXd& sigmaV);

	//for paper
	std::string stringRep;
	void outputForPaper(std::ofstream& file);
	double bornTime;

	double firstSize() { return o_firstSize; }
	double lastSize() { return o_lastSize; }
	double firstGFP() { return o_firstGFP; }
	double lastGFP() { return o_lastGFP; }
	double meanGFP() { return o_meanGFP; }

private:
	int o_left = -1; //children. -1 means NULL
	int o_right = -1;
	bool o_isLeaf;
	std::vector<NodeLDS>* o_tree;
	int o_parent = -1;
	int updatedChild = 0; //used in filtering recursion
	double o_logc = 1.0; //log c in my note on overleaf
	Eigen::MatrixXd syn_var;  //calculated in filtering, and used in smoothing

	double o_firstSize = 0;
	double o_lastSize = 0;
	double o_firstGFP = 0;
	double o_lastGFP = 0;
	double o_meanGFP = 0;

	//for debug
	double o_log_g = -1;
	double o_log_h = -1;

	

};

class TreeLDS
{
public:
	TreeLDS();
	~TreeLDS();

	friend class ForestLDS;

	NodeLDS& root() { return o_tree[0]; } //return root
	void load(std::string filename, int n, int m);
	void load_w(std::string filename, int n); //for wakamoto's data; to be deleted;
	double lambda; //growth rate
	int n; //dimension of hidden variables
	int m; //dimension of observed variables

	void smoothing(  //update root node's unconditioned distribution IN ADVANCE
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::MatrixXd& sigmaV
	);

	void smoothing(  //see my note on 21/5/2018
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::VectorXd& meanV,
		const Eigen::MatrixXd& sigmaV
	);

	void smoothing(
		Eigen::MatrixXd& A,
		Eigen::MatrixXd& C,
		Eigen::MatrixXd& sigmaW,
		Eigen::MatrixXd& sigmaV,
		Eigen::VectorXd& rootMean,
		Eigen::MatrixXd& rootVar);

	void smoothing(
		Eigen::MatrixXd& A,
		Eigen::MatrixXd& C,
		Eigen::MatrixXd& sigmaW,
		Eigen::VectorXd& meanV,
		Eigen::MatrixXd& sigmaV,
		Eigen::VectorXd& rootMean,
		Eigen::MatrixXd& rootVar
	);
	
	size_t size() { return o_tree.size(); }
	size_t leavesSize() { return leavesNo; }

	//report log-likelihood
	double logLik();

	//Mstep, update
	void update(  //smoothing MUST be called IN ADVANCE
		Eigen::MatrixXd& A, 
		Eigen::MatrixXd& C,
		Eigen::MatrixXd& sigmaW,
		Eigen::MatrixXd& sigmaV,
		Eigen::VectorXd& rootMean, 
		Eigen::MatrixXd& rootVar);

	void update(  //smoothing MUST be called IN ADVANCE
		Eigen::MatrixXd& A,
		Eigen::MatrixXd& C,
		Eigen::MatrixXd& sigmaW,
		Eigen::VectorXd& meanV,
		Eigen::MatrixXd& sigmaV,
		Eigen::VectorXd& rootMean,
		Eigen::MatrixXd& rootVar);

	//output
	void output(std::ofstream& file) {
		for (NodeLDS n : o_tree)
			if (!excludeRoot || !n.isroot())
				n.output(file);
	}

	void output() {
		std::ofstream out(defaultOutputFile);
		output(out);
	}

	void outputForPaper(std::ofstream& out) {
		for (auto n : o_tree)
			n.outputForPaper(out);
	}

	//for tree description
	//void MathematicaGraphics(std::string outPath);

	//for M-step, summation
	Eigen::MatrixXd zpzct;  // return average of E[z_k z_k'^t ], where k is a parent of k' for all transition
	Eigen::MatrixXd zpzpt;  // return average of E[z_p z_p^t] without root, where p is a parent
	Eigen::MatrixXd zczct; // return average of E[z_k z_k^t] without root
	Eigen::MatrixXd zkzkt;// return average of E[z_k z_k^t]
	Eigen::MatrixXd xkzkt; //return average of x_k * E[z_k^t]
	Eigen::MatrixXd xkxkt; //return average of x_k * x_k^t
	Eigen::VectorXd zk;  //return average of E[z_k]
	Eigen::VectorXd xk; //average of observed
	//for check
	Eigen::MatrixXd covarZpZc; // Covar(zp, zc), average
	Eigen::MatrixXd covarZ; //Covar(zp, zp), average
	double tptc; //E[tp tc]
	double tktk; //E[tk tk]
	double t; //E[t]
	double covarTpTc;
	double covarTkTk;
	void execSummation(); //compute all averages above
	void execSummation(const Eigen::MatrixXd& C, const Eigen::MatrixXd& sigmaV);

	//For Wakamoto's data
	void checkEstimatedResult();
	void samplePath(const Eigen::MatrixXd& C, const Eigen::MatrixXd& sigmaV);
	void outputSamplePath(std::ofstream& out);

	//for AIC selction
	std::vector<double> gentimes();

	//member aceess
	//std::vector<NodeLDS>& nodesVector() { return o_tree; }

	//swapping
	void swapGentime(const std::vector<double>& gentimes, const std::vector<int>& itr); //swap generation times GIVEN the random iterator sequence.

private:
	std::vector<NodeLDS> o_tree; //address of root node
	void recFoward(  //recursively calculate uncondition node of hidden variables //for notation, see my note
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::MatrixXd& sigmaV
	);

	void recFiltering(  //recursively calculate alpha  //for notation see my note
			const Eigen::MatrixXd& A,
			const Eigen::MatrixXd& C,
			const Eigen::MatrixXd& sigmaW,
			const Eigen::MatrixXd& sigmaV
			);

	void recSmoothing(
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::MatrixXd& sigmaV
	);

	void recFiltering(  //recursively calculate alpha  //for notation see my note
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::VectorXd& meanV,
		const Eigen::MatrixXd& sigmaV
	);

	
	std::vector<int> leaves;
	size_t leavesNo = 0; //number of leaves

	std::string defaultOutputFile;

};

double normalize(std::vector<double>& vec); // normalize and return the normalization factor

class ForestLDS
{
public:
	ForestLDS(int N, int M) : n(N), m(M) {};
	void add(TreeLDS* tree);
	size_t nodeSize() { size_t sum = 0; for (auto e : forest) sum += e->size(); return sum; }

	//dimensions
	int n;
	int m;

	void smoothing(  //update root node's unconditioned distribution IN ADVANCE
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::MatrixXd& sigmaV
	);
	void smoothing(  //see my note on 21/5/2018
		const Eigen::MatrixXd& A,
		const Eigen::MatrixXd& C,
		const Eigen::MatrixXd& sigmaW,
		const Eigen::VectorXd& meanV,
		const Eigen::MatrixXd& sigmaV
	);
	void smoothing(
		Eigen::MatrixXd& A,
		Eigen::MatrixXd& C,
		Eigen::MatrixXd& sigmaW,
		Eigen::MatrixXd& sigmaV,
		Eigen::VectorXd& rootMean,
		Eigen::MatrixXd& rootVar);
	void smoothing(
		Eigen::MatrixXd& A,
		Eigen::MatrixXd& C,
		Eigen::MatrixXd& sigmaW,
		Eigen::VectorXd& meanV,
		Eigen::MatrixXd& sigmaV,
		Eigen::VectorXd& rootMean,
		Eigen::MatrixXd& rootVar);

	//report log-likelihood
	double logLik();

	//Mstep, update
	void execSummation();
	void execSummation(const Eigen::MatrixXd& C, const Eigen::MatrixXd& sigmaV);
	void update(  //smoothing MUST be called IN ADVANCE
		Eigen::MatrixXd& A,
		Eigen::MatrixXd& C,
		Eigen::MatrixXd& sigmaW,
		Eigen::MatrixXd& sigmaV,
		Eigen::VectorXd& rootMean,
		Eigen::MatrixXd& rootVar);
	void update(  //smoothing MUST be called IN ADVANCE
		Eigen::MatrixXd& A,
		Eigen::MatrixXd& C,
		Eigen::MatrixXd& sigmaW,
		Eigen::VectorXd& meanV,
		Eigen::MatrixXd& sigmaV,
		Eigen::VectorXd& rootMean,
		Eigen::MatrixXd& rootVar);

	void output() { for (auto t : forest) t->output(); }

	//for debug
	void output(std::ofstream& out) { assert(!forest.empty()); forest[0]->output(out); }

	//for Wakamoto
	void checkEstimatedResult() { for (auto t : forest) t->checkEstimatedResult(); }	
	void samplePath(const Eigen::MatrixXd& C, const Eigen::MatrixXd& sigmaV) { for (auto t : forest) t->samplePath(C, sigmaV); }
	void outputSamplePath(std::ofstream& out) { for (auto t : forest) t->outputSamplePath(out); };

	//for paper
	void outputForPaper(std::ofstream& out) {
		for (auto t : forest)
			t->outputForPaper(out);
	}

	void outputForPaper(std::string filenamebase) {
		for (int i = 0; i != forest.size(); i++) {
			std::string filename_treeLabel = filenamebase + "_" + std::to_string(i) + ".txt";
			std::ofstream out(filename_treeLabel);
			forest[i]->outputForPaper(out);
		}
	}

	std::vector<TreeLDS*> forest;

	//summation
	Eigen::MatrixXd zpzct;
	Eigen::MatrixXd zpzpt;
	Eigen::MatrixXd zczct;
	Eigen::MatrixXd zkzkt;
	Eigen::MatrixXd xkzkt;
	Eigen::MatrixXd xkxkt;
	Eigen::MatrixXd covarZpZc;
	Eigen::MatrixXd covarZ;
	Eigen::MatrixXd rootVar;
	Eigen::MatrixXd rootMean;
	Eigen::VectorXd zk;
	Eigen::VectorXd xk;
	double tptc;
	double tktk;
	double t;
	double covarTkTk;
	double covarTpTc;
};