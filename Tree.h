#ifndef GUARD_HEDDER_TREE
#define GUARD_HEDDER_TREE

#include<vector>
#include<string>
#include "Density.h"
#include <functional>

class Tree;

class Node
{
public:
	Node(); //not for use
	Node::Node(double survivalTime, int typeno, int parentsID, std::vector<Node>* tree); //for internal nodes
	Node(int type, double survivalTime,int typeno, int parentsId, std::vector<Node>* tree); //for leaves
	~Node();

	int left() const { return o_left; } //return children's ID
	int right() const { return o_right; }
	int parent() const { return o_parent; } //return parent's ID
	int sister() const;  //return sister's ID
	double survivalTime; //return survival time
	double bornTime() { return o_borntime; }
	std::vector<double> currentDistribution;
	bool isLeaf() const { return o_isLeaf; }
	bool addChild(int childsID); //return whether successfully added the child. Left first, Right last.
	int type() const { return o_type; }
	void updateAlpha(const std::vector<std::vector<double>>& transitMat, const std::vector<Density*>& densities); //update alpha (see Note)
	bool updateBeta(const std::vector<std::vector<double>>& transitMat, const std::vector<Density*>& densities);
	std::vector<double> alpha;
	std::vector<double> beta;
	bool isleft() const; // return whether or not this is left child (root is false)
	bool isroot() const { return this == &o_tree->at(0); } //return whether this is root;
	double c() const { return o_c; } //return c; used to calculate log-likelihood
	void changeC(double c) { o_c = c; }
	std::string stringRep() { return o_stringrep; }

private:
	int o_left = -1; //children. -1 means NULL
	int o_right = -1;
	bool o_isLeaf;
	int o_type = -1;
	std::vector<Node>* o_tree;
	int o_parent = -1;
	int o_typeno() { return currentDistribution.size(); }
	bool betaUpdateFlag = true;
	double o_c = 1.0; //for definition, see not on the 21 June 2017
	double o_borntime = 0.0;
	std::string o_stringrep = "";
};


class Tree
{
public:
	Tree();
	~Tree();
	Tree(std::vector<double> steadyDistribution, double lambda) : steadyDistribution(steadyDistribution), lambda(lambda) {}
	Node& root() { return o_tree[0]; } //return root
	void load(std::string filename);
	void load(int typeNo, std::string filename, bool paperFlag = false);  //load file w/o head information, i.e. (lines, typeno)
	std::vector<double> steadyDistribution;
	double lambda; //growth rate
	int typeNo() const { return typeno; }

	//update all o_currentDistribution. transit[i][j] means the prob. of i -> j
	void inference(const std::vector<std::vector<double>>& transitMat, const std::vector<Density*>& densities);

	//for data generation
	void steadyState(double experimentTime);

	//for M-step, summation
	std::vector<double> sum(std::function<double(const Node&, int)> f); //j-th element of return is \sum f(node, j) for all node i. f(node, type)
	std::vector<std::vector<double>> transitCoef(const std::vector<std::vector<double>>& transit, const std::vector<Density*>& densities); //return the coefficient of 1 / T[i][j] in the derivative of M-step (i.e. averaged likelyhood).

	//for exception
	size_t size() { return o_tree.size(); }
	size_t leavesSize() { return leavesNo; }

	//for root and leaves' EM for age
	//std::vector<double> rootLeavesEM(double& sum, double& weightSub, const std::vector<double> integrated, const double infValue); // 

	//report log-likelihood
	double logLik(bool leafIncluded = true);

	//for tree description
	void MathematicaGraphics(std::string outPath);


private:
	std::vector<Node> o_tree; //address of root node
	void calcAlpha(const std::vector<std::vector<double>>& transitMat, const std::vector<Density*>& densities); //recursively calculate alpha
	void calcBeta(const std::vector<std::vector<double>>& transitMat, const std::vector<Density*>& densities);
	int typeno;
	std::vector<int> leaves;
	size_t leavesNo = 0; //number of leaves
};

double normalize(std::vector<double>& vec); // normalize and return the normalization factor


#endif