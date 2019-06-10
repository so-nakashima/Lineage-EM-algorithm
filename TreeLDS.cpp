#include "TreeLDS.h"
#include "assert.h"
#include <fstream>
#include <iostream>
#include <queue>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/src/Core/Inverse.h>
#include <Eigen/Dense>
#include <map>
#include <limits>
#include "LDSestimator.h"
#include "dataGenerator.h"

using namespace Eigen;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;

// fast_inverse http://tenteroring.luna.ddns.vc/tenteroring/2014/11/942
template<typename T>
void fast_inverse(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& dst) {
	switch (m.rows()) {
	case 2: {
		Eigen::Matrix<T, 2, 2> hoge(m);
		dst = hoge.inverse();
		break;
	}
	case 3: {
		Eigen::Matrix<T, 3, 3> hoge(m);
		dst = hoge.inverse();
		break;
	}
	case 4: {
		Eigen::Matrix<T, 4, 4> hoge(m);
		dst = hoge.inverse();
		break;
	}
	default:
		dst = m.inverse();
		break;
	}
}
//void fast_inverse(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& m, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& dst) {
//	switch (m.rows()) {
//	case 2:
//		dst = internal::inverse_impl<Eigen::Matrix<double, 2, 2>>(m);
//		break;
//	case 3:
//		dst = Eigen::internal::inverse_impl<Eigen::Matrix<double, 3, 3>>(m);
//		break;
//	case 4:
//		dst = Eigen::internal::inverse_impl<Eigen::Matrix<double, 4, 4>>(m);
//		break;
//	default:
//		dst = m.inverse();
//		break;
//	}
//}


NodeLDS::NodeLDS()
{
}

NodeLDS::NodeLDS(VectorXd ArgObserved, int parentsID, bool isLeaf, int N, int M, std::vector<NodeLDS>* tree) :
	observed(ArgObserved), o_isLeaf(isLeaf), o_tree(tree), o_parent(parentsID), n(N), m(M) {
	assert(ArgObserved.size() == m);
	
	//initializations of smoothed and filtered distributions 
	smoothe_mean =  filter_mean = VectorXd(n);
	smoothe_var = filter_var = MatrixXd(n, n);
	predObserved = Vec::Zero(n);

	//for paper
	if (parentsID >= 0) {
		NodeLDS& parentNode = (*tree)[parentsID];
		stringRep = parentNode.stringRep + std::to_string(parentNode.children.size() + 1);
		bornTime = parentNode.bornTime + exp(parentNode.observed[0]);
	}
	else {
		stringRep = "1";
		bornTime = 0.0;
	}
}

NodeLDS::NodeLDS(Eigen::VectorXd ArgObserved, int parentsID, bool isLeaf, int N, int M, double firstSize, double lastSize, double firstGFP, double lastGFP, double meanGFP, std::vector<NodeLDS>* tree) :
	observed(ArgObserved), o_isLeaf(isLeaf), o_tree(tree), o_parent(parentsID), n(N), m(M), o_firstSize(firstSize), o_lastSize(lastSize), o_firstGFP(firstGFP), o_lastGFP(lastGFP), o_meanGFP(meanGFP) {
	assert(ArgObserved.size() == m);

	//initializations of smoothed and filtered distributions 
	smoothe_mean = filter_mean = VectorXd(n);
	smoothe_var = filter_var = MatrixXd(n, n);
	predObserved = Vec::Zero(n);

	//for paper
	if (parentsID >= 0) {
		NodeLDS& parentNode = (*tree)[parentsID];
		stringRep = parentNode.stringRep + std::to_string(parentNode.children.size() + 1);
		bornTime = parentNode.bornTime + exp(parentNode.observed[0]);
	}
	else {
		stringRep = "1";
		bornTime = 0.0;
	}
}

//bool NodeLDS::isleft() const {
//	if (isroot())
//		return false;
//	NodeLDS parentNodeLDS = (*o_tree)[parent()];
//	return this == &((*o_tree)[parentNodeLDS.left()]);
//}

NodeLDS::~NodeLDS()
{
}

//int NodeLDS::sister() const {
//	if (isroot())  //except root case
//		return -1;
//
//	int parentID = parent();
//	int leftID = (*o_tree)[parentID].left();
//	int rightID = (*o_tree)[parentID].right();
//	if (isleft())
//		return rightID;
//	else
//		return leftID;
//}

void NodeLDS::addChild(int childId) {
	assert(!isLeaf());
	children.push_back(childId);
}

TreeLDS::TreeLDS()
{
}

TreeLDS::~TreeLDS()
{
}

void TreeLDS::load(std::string filename, int N, int M) {  
	std::ifstream file(filename.c_str());
	if (!file) {
		std::cout << "Cannot open \"" << filename << "\"." << std::endl;
		return;
	}

	//initialize tree n and m
	n = N; m = M;

	//load nodes
	o_tree.clear();
	int parent;
	int count = 0;
	while (file >> parent) {
		int isleaf; file >> isleaf;

		VectorXd observed(m);
		for (int i = 0; i != m; i++) {
			double temp;
			file >> temp;
			observed[i] = temp;
		}
		
		o_tree.push_back(NodeLDS(observed, parent, isleaf, n, m, &o_tree));
		if (isleaf) {
			leaves.push_back(count);
			//increment the cardinality of leaves
			leavesNo++;
		}

		
		if (count != 0) {
			o_tree[parent].addChild(count);
		}

		count++;
	}

	defaultOutputFile = filename + "_smoothed.txt";
}

void TreeLDS::load_w(std::string filename, int N) {
	std::ifstream file(filename.c_str());
	if (!file) {
		std::cout << "Cannot open \"" << filename << "\"." << std::endl;
		return;
	}

	//initialize tree n and m
	n = N; m = 1;

	//remember parents true location in o_tree
	std::map<int, int> line2id;

	//load nodes
	o_tree.clear();
	int parent;
	int count = 0;
	std::string temp;
	double firstArea, lastArea, firstGFP, lastGFP, meanGFP;
	while (file >> parent) {
		int isleaf; file >> isleaf;

		double survivalTime;
		file >> temp >> survivalTime >> firstArea >> lastArea >> firstGFP >> lastGFP >> meanGFP;// >> temp >> temp;  //trash irrelevant information
		
		VectorXd observed(m);
		observed[0] = log(survivalTime);

		
		int parentID = parent;
		if (count != 0) {
			assert(line2id.find(parent) != line2id.end());
			parentID = line2id[parent];
		}

		//trash leaf information
		if (!isleaf) {
			o_tree.push_back(NodeLDS(observed, parentID, isleaf, n, m, firstArea, lastArea, firstGFP, lastGFP, meanGFP, &o_tree));
			line2id[count] = o_tree.size() - 1;
		}

		if (count != 0 && !isleaf) {
			o_tree[parentID].addChild(o_tree.size() - 1);
		}

		//Update count even when this node had been trashed;
		count++;
	}

	//re-assign isLeaf
	for (int i = 0; i != o_tree.size(); i++) {
		if (o_tree[i].children.size() == 0) {
			o_tree[i].isLeaf() = true;
			leaves.push_back(i);
			leavesNo++;
		}
	}

	defaultOutputFile = filename + "_smoothed.txt";
}

double TreeLDS::logLik() {
	if (excludeRoot) {
		double sum = 0.0;
		for (int cID : root().children)
			sum += o_tree[cID].log_c();
		return sum;
	}
	else
		return root().log_c();
}

void TreeLDS::smoothing(
	Eigen::MatrixXd& A,
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar
) {
	assert(rootMean.size() == A.rows());
	assert(rootVar.cols() == rootVar.rows());
	assert(rootVar.cols() == A.rows());

	NodeLDS& rootNode = root();
	if (excludeRoot) {
		for (int cID : rootNode.children) {
			NodeLDS& child = o_tree[cID];
			child.forward_mean = rootMean;
			child.forward_var = rootVar;
		}
	}
	else {
		rootNode.forward_mean = rootMean;
		rootNode.forward_var = rootVar;
	}
	smoothing(A, C, sigmaW, sigmaV);
}

void TreeLDS::smoothing(
	Eigen::MatrixXd& A,
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::VectorXd& meanV,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar
) {
	assert(rootMean.size() == A.rows());
	assert(rootVar.cols() == rootVar.rows());
	assert(rootVar.cols() == A.rows());

	NodeLDS& rootNode = root();
	if (excludeRoot) {
		for (int cID : rootNode.children) {
			NodeLDS& child = o_tree[cID];
			child.forward_mean = rootMean;
			child.forward_var = rootVar;
		}
	}
	else {
		rootNode.forward_mean = rootMean;
		rootNode.forward_var = rootVar;
	}
	smoothing(A, C, sigmaW, meanV, sigmaV);
}

void NodeLDS::output(std::ofstream& file) {
	for (int i = 0; i != filter_mean.size(); i++)
		file << filter_mean[i] << " ";
	for (int i = 0; i != filter_var.rows(); i++)
		for (int j = 0; j != filter_var.cols(); j++)
			file << filter_var(i, j) << " ";
	for (int i = 0; i != smoothe_mean.size(); i++)
		file << smoothe_mean[i] << " ";
	for (int i = 0; i != smoothe_var.rows(); i++)
		for (int j = 0; j != smoothe_var.cols(); j++)
			file << smoothe_var(i, j) << " ";
	file << o_log_g << " " << o_log_h << " ";
	for (int i = 0; i != predObserved.size(); i++)
		file << predObserved[i] << " ";
	file << std::endl;
}

void TreeLDS::smoothing(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::MatrixXd& sigmaV
) {
	//assertions
	assert(A.rows() == n && A.cols() == n);
	assert(C.rows() == m && C.cols() == n);
	assert(sigmaW.rows() == n && sigmaW.cols() == n);
	assert(sigmaV.rows() == m && sigmaV.cols() == m);

	//fowrd recursion to calculate p(z_k)
	recFoward(A, C, sigmaW, sigmaV);

	//recursive filtering and smoothing
	recFiltering(A, C, sigmaW, sigmaV);
	recSmoothing(A, C, sigmaW, sigmaV);
}

void TreeLDS::smoothing(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::VectorXd& meanV,
	const Eigen::MatrixXd& sigmaV
) {
	//assertions
	assert(A.rows() == n && A.cols() == n);
	assert(C.rows() == m && C.cols() == n);
	assert(sigmaW.rows() == n && sigmaW.cols() == n);
	assert(sigmaV.rows() == m && sigmaV.cols() == m);

	//fowrd recursion to calculate p(z_k)
	recFoward(A, C, sigmaW, sigmaV);

	//recursive filtering and smoothing
	recFiltering(A, C, sigmaW, meanV, sigmaV);
	recSmoothing(A, C, sigmaW, sigmaV);
}

void TreeLDS::recFoward(  
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::MatrixXd& sigmaV
) {

	//recursive computation
	std::queue<int> q;
	for (const auto& e : root().children)
		q.push(e);

	while (!q.empty()) {
		NodeLDS& current = o_tree[q.front()];
		q.pop();
		if(!excludeRoot || current.parent() != 0)
			current.updateFoward(A, C, sigmaW, sigmaV);

		//for children
		if (!current.isLeaf()) {
			for (const auto& e : current.children)
				q.push(e);
		}
	}
}

void NodeLDS::updateFoward(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::MatrixXd& sigmaV
) {
	NodeLDS& parentNode = (*o_tree)[parent()];
	forward_mean = A * parentNode.forward_mean;
	forward_var = sigmaW + A * parentNode.forward_var * A.transpose();
}

void TreeLDS::recFiltering(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::MatrixXd& sigmaV
)
{
	//recursive computation
	std::queue<int> q;
	for (int i = 0; i != leaves.size(); i++) {
		q.push(leaves[i]);
	}
	while (!q.empty()) {
		NodeLDS& current = o_tree[q.front()];
		q.pop();
		//update
		bool flag = current.updateFilter(A, C, sigmaW, sigmaV);
		//recursion if update sucessed 
		bool recursionFlag = flag && ((excludeRoot && current.parent() != 0) || (!excludeRoot && !current.isroot()));
		if (recursionFlag)
			q.push(current.parent());
	}
}

void TreeLDS::recFiltering(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::VectorXd& meanV,
	const Eigen::MatrixXd& sigmaV
)
{
	//recursive computation
	std::queue<int> q;
	for (int i = 0; i != leaves.size(); i++) {
		q.push(leaves[i]);
	}
	while (!q.empty()) {
		NodeLDS& current = o_tree[q.front()];
		q.pop();
		//update
		bool flag = current.updateFilter(A, C, sigmaW, meanV, sigmaV);
		//recursion if update sucessed 
		bool recursionFlag = flag && ((excludeRoot && current.parent() != 0) || (!excludeRoot && !current.isroot()));
		if (recursionFlag)
			q.push(current.parent());
	}
}

double log_normal_pdf(Vec x, Vec mean, Mat covar) { //log. of normal distribution
	assert(mean.size() == covar.rows());
	assert(mean.size() == covar.cols());
	assert(x.size() == mean.size());

	int n = mean.size();
	Vec diff = x - mean;
	Mat covar_inv; fast_inverse(covar, covar_inv);

	double exponent = diff.transpose() * covar_inv * diff;

	return - 0.5 * exponent - n * 0.5 * log(2 * 3.141592) - 0.5 * log(covar.determinant()) ;
}

bool NodeLDS::updateFilter(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::MatrixXd& sigmaV
) {
	//treat leaves separately
	if (isLeaf()) {
		//computation
		Mat funi = C * forward_var * C.transpose() + sigmaV;
		Mat funi_inv; fast_inverse(funi, funi_inv);
		Mat K = forward_var * C.transpose() * funi_inv;
		filter_mean = forward_mean + K * (observed - C * forward_mean);
		filter_var = forward_var - K * C * forward_var;

		//update syn_var  //for smoothing
		syn_var = forward_var;

		//update c
		o_logc = log_normal_pdf(observed, C * forward_mean, C * forward_var * C.transpose() + sigmaV);

		//for debug
		o_log_g = o_logc;
		Cfilter = C * filter_mean;
		forcasted = C * forward_mean;

		return true;  //successfully updated
	}

	//bulk cells
	//update start if two children are updated
	updatedChild++;
	if (updatedChild != children.size())
		return false; //not updated

	//computation
	//1step prediction
	Mat forward_inv; fast_inverse(forward_var, forward_inv);
	Mat sigmaW_inv; fast_inverse(sigmaW, sigmaW_inv);
	Mat Q_inv = forward_inv + A.transpose() * sigmaW_inv * A;
	Mat Q; fast_inverse(Q_inv, Q);

	Mat sigma_pred_inv_sum = Eigen::MatrixXd::Zero(A.rows(), A.cols()); //justified by assertion in TreeLDS::smoothe
	Vec mu_pred_sum = Eigen::VectorXd::Zero(A.rows());

	double exponent_sum = 0;  //for the calculation of c (i.e. likelihood)
	double log_determinant_sum = 0;
	double logc_children_sum = 0; //c of children  //log

	for (int id : children) {
		NodeLDS& current = (*o_tree)[id];

		Vec mu = Q * (A.transpose() * sigmaW_inv * current.filter_mean + forward_inv * forward_mean);
		Mat hoge = Q * A.transpose() * sigmaW_inv;
		Mat sigma = Q + hoge * current.filter_var * hoge.transpose();
		
		Mat sigma_inv; fast_inverse(sigma, sigma_inv);
		sigma_pred_inv_sum += sigma_inv;
		mu_pred_sum += sigma_inv * mu;

		exponent_sum += mu.transpose() * sigma_inv * mu;
		log_determinant_sum += log(sigma.determinant());

		logc_children_sum += current.log_c();
	}

	//synthesize
	Mat sigma_syn_inv = sigma_pred_inv_sum - (children.size() - 1) * forward_inv;
	Mat sigma_syn; fast_inverse(sigma_syn_inv, sigma_syn);
	Vec mu_syn = sigma_syn * (mu_pred_sum - (children.size() - 1 ) * forward_inv * forward_mean);

	double exponent_syn = mu_syn.transpose() * sigma_syn_inv * mu_syn; 
	double exponent_forward = forward_mean.transpose() * forward_inv * forward_mean;
	double log_h = 0.5 * (
		log(sigma_syn.determinant()) - log_determinant_sum + (children.size() - 1) * log(forward_var.determinant())
		+ exponent_syn + (children.size() - 1) * exponent_forward - exponent_sum
		); //my note of overleaf

	//update by observation
	Mat funi = sigmaV + C * sigma_syn * C.transpose();
	Mat funi_inv; fast_inverse(funi, funi_inv);
	Mat K = sigma_syn * C.transpose() * funi_inv;
	filter_mean = mu_syn + K * (observed - C * mu_syn);
	filter_var = sigma_syn - K * C * sigma_syn;

	double log_g = log_normal_pdf(observed, C * mu_syn, sigmaV + C * sigma_syn * C.transpose()); //my note of overleaf

	//update syn_var
	syn_var = sigma_syn;

	//update c
	o_logc = logc_children_sum + log_g + log_h;
	if (isnan(o_logc))
		A.inverse();

	//for debug
	o_log_g = log_g;
	o_log_h = log_h;
	Cfilter = C * filter_mean;
	forcasted = C * mu_syn;
	
	//notify that update suceeded
	updatedChild = 0;
	return true;
}

bool NodeLDS::updateFilter(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::VectorXd& meanV,
	const Eigen::MatrixXd& sigmaV
) {
	Eigen::VectorXd diff = observed - meanV;

	//treat leaves separately
	if (isLeaf()) {
		//computation
		Mat funi = C * forward_var * C.transpose() + sigmaV;
		Mat funi_inv; fast_inverse(funi, funi_inv);
		Mat K = forward_var * C.transpose() * funi_inv;
		filter_mean = forward_mean + K * (diff - C * forward_mean);
		filter_var = forward_var - K * C * forward_var;

		//update syn_var  //for smoothing
		syn_var = forward_var;

		//update c
		o_logc = log_normal_pdf(diff, C * forward_mean, C * forward_var * C.transpose() + sigmaV);

		//for debug
		o_log_g = o_logc;
		Cfilter = C * filter_mean;
		forcasted = C * forward_mean;

		return true;  //successfully updated
	}

	//bulk cells
	//update start if two children are updated
	updatedChild++;
	if (updatedChild != children.size())
		return false; //not updated

					  //computation
					  //1step prediction
	Mat forward_inv; fast_inverse(forward_var, forward_inv);
	Mat sigmaW_inv; fast_inverse(sigmaW, sigmaW_inv);
	Mat Q_inv = forward_inv + A.transpose() * sigmaW_inv * A;
	Mat Q; fast_inverse(Q_inv, Q);

	Mat sigma_pred_inv_sum = Eigen::MatrixXd::Zero(A.rows(), A.cols()); //justified by assertion in TreeLDS::smoothe
	Vec mu_pred_sum = Eigen::VectorXd::Zero(A.rows());

	double exponent_sum = 0;  //for the calculation of c (i.e. likelihood)
	double log_determinant_sum = 0;
	double logc_children_sum = 0; //c of children  //log

	for (int id : children) {
		NodeLDS& current = (*o_tree)[id];

		Vec mu = Q * (A.transpose() * sigmaW_inv * current.filter_mean + forward_inv * forward_mean);
		Mat hoge = Q * A.transpose() * sigmaW_inv;
		Mat sigma = Q + hoge * current.filter_var * hoge.transpose();

		Mat sigma_inv; fast_inverse(sigma, sigma_inv);
		sigma_pred_inv_sum += sigma_inv;
		mu_pred_sum += sigma_inv * mu;

		exponent_sum += mu.transpose() * sigma_inv * mu;
		log_determinant_sum += log(sigma.determinant());

		logc_children_sum += current.log_c();
	}

	//synthesize
	Mat sigma_syn_inv = sigma_pred_inv_sum - (children.size() - 1) * forward_inv;
	Mat sigma_syn; fast_inverse(sigma_syn_inv, sigma_syn);
	Vec mu_syn = sigma_syn * (mu_pred_sum - (children.size() - 1) * forward_inv * forward_mean);

	double exponent_syn = mu_syn.transpose() * sigma_syn_inv * mu_syn;
	double exponent_forward = forward_mean.transpose() * forward_inv * forward_mean;
	double log_h = 0.5 * (
		log(sigma_syn.determinant()) - log_determinant_sum + (children.size() - 1) * log(forward_var.determinant())
		+ exponent_syn + (children.size() - 1) * exponent_forward - exponent_sum
		); //my note of overleaf

		   //update by observation
	Mat funi = sigmaV + C * sigma_syn * C.transpose();
	Mat funi_inv; fast_inverse(funi, funi_inv);
	Mat K = sigma_syn * C.transpose() * funi_inv;
	filter_mean = mu_syn + K * (diff - C * mu_syn);
	filter_var = sigma_syn - K * C * sigma_syn;

	double log_g = log_normal_pdf(diff, C * mu_syn, sigmaV + C * sigma_syn * C.transpose()); //my note of overleaf

																								 //update syn_var
	syn_var = sigma_syn;

	//update c
	o_logc = logc_children_sum + log_g + log_h;
	if (isnan(o_logc))
		A.inverse();

	//for debug
	o_log_g = log_g;
	o_log_h = log_h;
	Cfilter = C * filter_mean;
	forcasted = C * mu_syn;

	//notify that update suceeded
	updatedChild = 0;
	return true;
}


void TreeLDS::recSmoothing(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::MatrixXd& sigmaV
){
	//root's smoothing is identical to filtering
	if (excludeRoot) {
		for (int cID : root().children) {
			NodeLDS& child = o_tree[cID];
			child.smoothe_mean = child.filter_mean;
			child.smoothe_var = child.filter_var;
			child.predObserved = C * child.smoothe_mean;
		}
	}
	else {
		root().smoothe_mean = root().filter_mean;
		root().smoothe_var = root().filter_var;
		root().predObserved = C * root().smoothe_mean;
	}

	//recursive computation
	std::queue<int> q;
	for (const auto& e : root().children) 
		q.push(e);

	while (!q.empty()) {
		NodeLDS& current = o_tree[q.front()];
		q.pop();
		if(!excludeRoot || current.parent() != 0)
			current.updateSmoothe(A, C, sigmaW, sigmaV);

		//for children
		if (!current.isLeaf()) {
			for (const auto& e : current.children)
				q.push(e);
		}
	}
}

void NodeLDS::updateSmoothe(
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::MatrixXd& sigmaV
) {
	NodeLDS& parentNode = (*o_tree)[parent()];

	Mat parent_forward_inv; fast_inverse(parentNode.forward_var, parent_forward_inv);
	Mat sigmaW_inv; fast_inverse(sigmaW, sigmaW_inv);
	Mat Q_inv = parent_forward_inv + A.transpose() * sigmaW_inv * A;
	Mat Q; fast_inverse(Q_inv, Q);
	Vec mu_pred = Q * (A.transpose() * sigmaW_inv * filter_mean + parent_forward_inv * parentNode.forward_mean);
	Mat predMat = Q * A.transpose() * sigmaW_inv;
	Mat sigma = Q + predMat * filter_var * predMat.transpose();
	Mat sigma_inv; fast_inverse(sigma, sigma_inv);

	J = filter_var * predMat.transpose() * sigma_inv;
	smoothe_mean = filter_mean + J * (parentNode.smoothe_mean - mu_pred);
	smoothe_var = filter_var + J * (parentNode.smoothe_var - sigma) * J.transpose();  //syn_var -> sigma ???

	predObserved = C * smoothe_mean;
}



void TreeLDS::update(Mat& A, Mat& C, Mat& sigmaW, Mat& sigmaV, Vec& rootMean, Mat& rootVar) {

	const double eta = 1.0;  //learning rate

	//assertions
	assert(A.rows() == n && A.cols() == n);
	assert(C.rows() == m && C.cols() == n);
	assert(sigmaW.rows() == n && sigmaW.cols() == n);
	assert(sigmaV.rows() == m && sigmaV.cols() == m);

	//initilize root distribution 
	NodeLDS& rootNode = root();

	//summation for M-step
	execSummation();

	//inverse matrices
	Mat zpzpt_inv; fast_inverse(zpzpt, zpzpt_inv);
	Mat zkzkt_inv; fast_inverse(zkzkt, zkzkt_inv);

	//update
	if (excludeRoot) {
		Vec rootMeanSum = Vec::Zero(n);
		Mat rootVarSum = Mat::Zero(n, n);
		double N = root().children.size();
		for (int cID : root().children) {
			rootMeanSum += o_tree[cID].smoothe_mean / N;
			rootVarSum += o_tree[cID].smoothe_var / N;
		}
		rootMean = (1.0 - eta) * rootMean + eta * rootMeanSum;
		rootVar = (1.0 - eta) * rootVar + eta * rootVarSum;
	}
	else {
		rootMean = (1.0 - eta) * rootMean + eta * rootNode.smoothe_mean;
		rootVar = (1.0 - eta) * rootVar + eta * rootNode.smoothe_var;
	}
	A = (1.0 - eta) * A + eta * zpzct.transpose() * zpzpt_inv;
	C = (1.0 - eta) * C + eta * xkzkt * zkzkt_inv;
	//sigmaW = (1.0 - eta) * sigmaW + eta * (zczct - A * zpzct - zpzct.transpose() * A.transpose() + A * zpzpt * A.transpose()); // the latter half is redundant. Substitute A_new by its definition. However, this redundant expression stabilize the positivity of sigmas
	//sigmaV = (1.0 - eta) * sigmaV + eta * (xkxkt - C * xkzkt.transpose() - xkzkt * C.transpose() + C * zkzkt * C.transpose());  //the same as above
	sigmaW = zczct - A * zpzct - zpzct.transpose() * A.transpose() + A * zpzpt * A.transpose();
	sigmaV = xkxkt - C * xkzkt.transpose() - xkzkt * C.transpose() + C * zkzkt * C.transpose();

	//symmetrize
	symmetrize(sigmaW); symmetrize(sigmaV); symmetrize(rootVar);
}

void gaugeTransformation(Mat& L, Mat& A, Mat& C, Mat& sigmaW, Vec& rootMean, Mat& rootVar) {
	Mat Linv; fast_inverse(L, Linv);

	//transformation
	A = L * A * Linv;
	//sigmaW = L * sigmaW * L.transpose();
	//C = C * Linv;
	rootMean = L * rootMean;
	rootVar = L * rootVar * L.transpose();
}

void TreeLDS::update(Mat& A, Mat& C, Mat& sigmaW, Vec& meanV, Mat& sigmaV, Vec& rootMean, Mat& rootVar) {

	const double eta = 1.0;  //learning rate

	//assertions
	assert(A.rows() == n && A.cols() == n);
	assert(C.rows() == m && C.cols() == n);
	assert(sigmaW.rows() == n && sigmaW.cols() == n);
	assert(sigmaV.rows() == m && sigmaV.cols() == m);

	//initilize root distribution 
	NodeLDS& rootNode = root();

	//summation for M-step
	execSummation();

	//inverse matrices
	Mat zpzpt_inv; fast_inverse(zpzpt, zpzpt_inv);
	Mat zkzkt_inv; fast_inverse(zkzkt, zkzkt_inv);

	//update
	if (excludeRoot) {
		Vec rootMeanSum = Vec::Zero(n);
		Mat rootVarSum = Mat::Zero(n, n);
		double N = root().children.size();
		for (int cID : root().children) {
			rootMeanSum += o_tree[cID].smoothe_mean / N;
			rootVarSum += o_tree[cID].smoothe_var / N;
		}
		rootMean = (1.0 - eta) * rootMean + eta * rootMeanSum;
		rootVar = (1.0 - eta) * rootVar + eta * rootVarSum;
	}
	else {
		rootMean = (1.0 - eta) * rootMean + eta * rootNode.smoothe_mean;
		rootVar = (1.0 - eta) * rootVar + eta * rootNode.smoothe_var;
	}
	A = (1.0 - eta) * A + eta * zpzct.transpose() * zpzpt_inv;
	C = (1.0 - eta) * C + eta * xkzkt * zkzkt_inv;
	//sigmaW = (1.0 - eta) * sigmaW + eta * (zczct - A * zpzct - zpzct.transpose() * A.transpose() + A * zpzpt * A.transpose()); // the latter half is redundant. Substitute A_new by its definition. However, this redundant expression stabilize the positivity of sigmas
	//sigmaV = (1.0 - eta) * sigmaV + eta * (xkxkt - C * xkzkt.transpose() - xkzkt * C.transpose() + C * zkzkt * C.transpose());  //the same as above
	sigmaW = zczct - A * zpzct - zpzct.transpose() * A.transpose() + A * zpzpt * A.transpose();

	meanV = xk - C * zk;
	Eigen::MatrixXd difdift = xkxkt - C * xkzkt.transpose() - xkzkt * C.transpose() + C * zkzkt * C.transpose();
	sigmaV = difdift - meanV * meanV.transpose();

	//diagonalize
	if (m == 1) {
		//svd i.e. diagonalization by eigenvalues
		JacobiSVD<MatrixXd> svd(sigmaW, ComputeFullU | ComputeFullV);
		Mat U = svd.matrixU();
		Mat D = svd.singularValues().asDiagonal();

		//normalize C
		Mat Cnew = C * U;
		Mat D2 = Eigen::MatrixXd::Zero(n, n);
		Mat D2inv = Eigen::MatrixXd::Zero(n, n);
		for (int i = 0; i != n; i++) {
			D2(i, i) = Cnew(1, i);
			D2inv(i, i) = Cnew(i, i);
			C(1, i) = 1.0;
		}
		Mat L = D2 * U.transpose();
		Mat Linv = U * D2inv;
		
		//transformation
		A = L * A * Linv;
		sigmaW = D2 * D * D2;
		rootMean = L * rootMean;
		rootVar = L * rootVar * L.transpose();
	}

	//symmetrize
	symmetrize(sigmaW); symmetrize(sigmaV); symmetrize(rootVar);
}

void TreeLDS::execSummation() {

	//initialization
	zpzct = MatrixXd::Zero(n,n);
	zpzpt = MatrixXd::Zero(n, n);
	zczct = MatrixXd::Zero(n, n);
	zkzkt = MatrixXd::Zero(n, n);
	xkzkt = MatrixXd::Zero(m, n);
	xkxkt = MatrixXd::Zero(m, m);
	zk = VectorXd::Zero(n);
	xk = VectorXd::Zero(m);
	covarZ = MatrixXd::Zero(n, n);
	covarZpZc = MatrixXd::Zero(n, n);
	tptc = 0;
	//Mat zkzkt_root = MatrixXd::Zero(n, n);
	//Mat zkzkt_leaves = MatrixXd::Zero(n, n);

	int N = size();
	if (excludeRoot)
		N--;

	//summation
	for (NodeLDS& current : o_tree) {
		if (excludeRoot && current.isroot())
			continue;

		//preparetion
		Mat current_zczct = (current.smoothe_var + current.smoothe_mean * current.smoothe_mean.transpose());

		//sum
		xkxkt += current.observed * current.observed.transpose() / N;
		xkzkt += current.observed * current.smoothe_mean.transpose() / N;
		zkzkt += current_zczct / N;
		zk += current.smoothe_mean / N;
		xk += current.observed / N;
		covarZ += current.smoothe_var;
		

		bool existParent = (excludeRoot && current.parent() != 0) || (!excludeRoot && !current.isroot());
		if(existParent) {
			//pareparation
			NodeLDS& parent = o_tree[current.parent()];
			Mat current_zpzct = parent.smoothe_var * current.J.transpose() + parent.smoothe_mean * current.smoothe_mean.transpose();
			Mat current_zpzpt = (parent.smoothe_var + parent.smoothe_mean * parent.smoothe_mean.transpose());

			//sum
			zpzpt += current_zpzpt / (N - 1);
			zpzct += current_zpzct / (N - 1);
			zczct += current_zczct / (N - 1);
			covarZpZc += parent.smoothe_var * current.J.transpose();
		}		
	}
}

void TreeLDS::execSummation(const Eigen::MatrixXd& C, const Eigen::MatrixXd& sigmaV) {

	//initialization
	zpzct = MatrixXd::Zero(n, n);
	zpzpt = MatrixXd::Zero(n, n);
	zczct = MatrixXd::Zero(n, n);
	zkzkt = MatrixXd::Zero(n, n);
	xkzkt = MatrixXd::Zero(m, n);
	xkxkt = MatrixXd::Zero(m, m);
	zk = VectorXd::Zero(n);
	covarZ = MatrixXd::Zero(n, n);
	covarZpZc = MatrixXd::Zero(n, n);
	tptc = 0;
	t = 0;
	covarTkTk = 0;
	covarTpTc = 0;
	//Mat zkzkt_root = MatrixXd::Zero(n, n);
	//Mat zkzkt_leaves = MatrixXd::Zero(n, n);

	int N = size();
	if (excludeRoot)
		N--;

	//summation
	for (NodeLDS& current : o_tree) {
		if (excludeRoot && current.isroot())
			continue;

		//preparetion
		Mat current_zczct = (current.smoothe_var + current.smoothe_mean * current.smoothe_mean.transpose());

		//sum
		xkxkt += current.observed * current.observed.transpose() / N;
		xkzkt += current.observed * current.smoothe_mean.transpose() / N;
		zkzkt += current_zczct / N;
		zk += current.smoothe_mean / N;
		covarZ += current.smoothe_var;
		tktk += (C * current.smoothe_var * C.transpose() + sigmaV + current.smoothe_mean.transpose() * C.transpose() * C * current.smoothe_mean)(0, 0) / N;
		t += exp((C * current.smoothe_mean + C * current.smoothe_var * C.transpose() * 0.5)(0, 0)) / N;

		bool existParent = (excludeRoot && current.parent() != 0) || (!excludeRoot && !current.isroot());
		if (existParent) {
			//pareparation
			NodeLDS& parent = o_tree[current.parent()];
			Mat current_zpzct = parent.smoothe_var * current.J.transpose() + parent.smoothe_mean * current.smoothe_mean.transpose();
			Mat current_zpzpt = (parent.smoothe_var + parent.smoothe_mean * parent.smoothe_mean.transpose());

			//sum
			zpzpt += current_zpzpt / (N - 1);
			zpzct += current_zpzct / (N - 1);
			zczct += current_zczct / (N - 1);
			covarZpZc += parent.smoothe_var * current.J.transpose();
			tptc += (C * parent.smoothe_var * current.J.transpose() * C.transpose() + (C * parent.smoothe_mean).transpose() * (C * current.smoothe_mean))(0,0) / (N - 1);
		}
	}
}

double ForestLDS::logLik() {
	double sum = 0.0;
	for (auto e : forest)
		sum += e->logLik();
	return sum;
}

void ForestLDS::smoothing(  //update root node's unconditioned distribution IN ADVANCE
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::MatrixXd& sigmaV
) {
	for (auto e : forest)
		e->smoothing(A, C, sigmaW, sigmaV);
}

void ForestLDS::smoothing(  //update root node's unconditioned distribution IN ADVANCE
	const Eigen::MatrixXd& A,
	const Eigen::MatrixXd& C,
	const Eigen::MatrixXd& sigmaW,
	const Eigen::VectorXd& meanV,
	const Eigen::MatrixXd& sigmaV
) {
	for (auto e : forest)
		e->smoothing(A, C, sigmaW, meanV, sigmaV);
}

void ForestLDS::smoothing(
	Eigen::MatrixXd& A,
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar
) {
	//initialize rootMean and rootVar
	for (auto e : forest) {
		NodeLDS& rootNode = e->root();
		if (excludeRoot) {
			for (int cID : rootNode.children) {
				NodeLDS& child = e->o_tree[cID];
				child.forward_mean = rootMean;
				child.forward_var = rootVar;
			}
		}
		else {
			rootNode.forward_mean = rootMean;
			rootNode.forward_var = rootVar;
		}
	}

	smoothing(A, C, sigmaW, sigmaV);
}

void ForestLDS::smoothing(
	Eigen::MatrixXd& A,
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::VectorXd& meanV,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& rootMean,
	Eigen::MatrixXd& rootVar
) {
	//initialize rootMean and rootVar
	for (auto e : forest) {
		NodeLDS& rootNode = e->root();
		if (excludeRoot) {
			for (int cID : rootNode.children) {
				NodeLDS& child = e->o_tree[cID];
				child.forward_mean = rootMean;
				child.forward_var = rootVar;
			}
		}
		else {
			rootNode.forward_mean = rootMean;
			rootNode.forward_var = rootVar;
		}
	}

	smoothing(A, C, sigmaW, meanV, sigmaV);
}

void ForestLDS::add(TreeLDS* hoge) {
	assert(hoge->n == n && hoge->m == m);
	forest.push_back(hoge);
}

void ForestLDS::execSummation() {
	//initialization
	zpzct = MatrixXd::Zero(n, n);
	zpzpt = MatrixXd::Zero(n, n);
	zczct = MatrixXd::Zero(n, n);
	zkzkt = MatrixXd::Zero(n, n);
	xkzkt = MatrixXd::Zero(m, n);
	xkxkt = MatrixXd::Zero(m, m);
	rootVar = MatrixXd::Zero(n, n);
	rootMean = VectorXd::Zero(n);
	covarZ = MatrixXd::Zero(n, n);
	covarZpZc = MatrixXd::Zero(n, n);
	zk = VectorXd::Zero(n);
	xk = VectorXd::Zero(m);

	//summation
	int N = nodeSize();
	for (auto e : forest) {
		double frac = e->size() / (double)N;
		e->execSummation();
		zpzct += e->zpzct * frac;
		zpzpt += e->zpzpt * frac;
		zczct += e->zczct * frac;
		zkzkt += e->zkzkt * frac;
		xkzkt += e->xkzkt * frac;
		xkxkt += e->xkxkt * frac;
		covarZ += e->covarZ * frac;
		covarZpZc += e->covarZpZc * frac;
		zk += e->zk * frac;
		xk += e->xk * frac;

		//root Mean and root Var
		if (excludeRoot) {
			double childrenNo = e->root().children.size();
			for (int cID : e->root().children) {
				NodeLDS& child = e->o_tree[cID];
				rootMean += child.smoothe_mean / childrenNo / forest.size();
				rootVar += child.smoothe_var / childrenNo / forest.size();
			}
		}
		else {
			rootMean += e->root().smoothe_mean / forest.size();
			rootVar += e->root().smoothe_var / forest.size();
		}
	}

}

void ForestLDS::execSummation(const Eigen::MatrixXd& C, const Eigen::MatrixXd& sigmaV) {
	//initialization
	zpzct = MatrixXd::Zero(n, n);
	zpzpt = MatrixXd::Zero(n, n);
	zczct = MatrixXd::Zero(n, n);
	zkzkt = MatrixXd::Zero(n, n);
	xkzkt = MatrixXd::Zero(m, n);
	xkxkt = MatrixXd::Zero(m, m);
	rootVar = MatrixXd::Zero(n, n);
	rootMean = VectorXd::Zero(n);
	covarZ = MatrixXd::Zero(n, n);
	covarZpZc = MatrixXd::Zero(n, n);
	zk = VectorXd::Zero(n);
	xk = VectorXd::Zero(m);
	tktk = 0;
	tptc = 0;
	t = 0;

	//summation
	int N = nodeSize();
	for (auto e : forest) {
		double frac = e->size() / (double)N;
		e->execSummation(C, sigmaV);
		zpzct += e->zpzct * frac;
		zpzpt += e->zpzpt * frac;
		zczct += e->zczct * frac;
		zkzkt += e->zkzkt * frac;
		xkzkt += e->xkzkt * frac;
		xkxkt += e->xkxkt * frac;
		covarZ += e->covarZ * frac;
		covarZpZc += e->covarZpZc * frac;
		zk += e->zk * frac;
		xk += e->xk * frac;
		tktk += e->tktk * frac;
		tptc += e->tptc * frac;
		t += e->t * frac;

		//root Mean and root Var
		if (excludeRoot) {
			double childrenNo = e->root().children.size();
			for (int cID : e->root().children) {
				NodeLDS& child = e->o_tree[cID];
				rootMean += child.smoothe_mean / childrenNo / forest.size();
				rootVar += child.smoothe_var / childrenNo / forest.size();
			}
		}
		else {
			rootMean += e->root().smoothe_mean / forest.size();
			rootVar += e->root().smoothe_var / forest.size();
		}
	}

}

void ForestLDS::update(  //smoothing MUST be called IN ADVANCE
	Eigen::MatrixXd& A,
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& argRootMean,
	Eigen::MatrixXd& argRootVar
) {
	const double eta = 1.0; //learning rate

	//summation
	execSummation();


	//update
	argRootMean = rootMean;
	argRootVar = rootVar;

	Mat zpzpt_inv;  fast_inverse(zpzpt, zpzpt_inv);
	Mat zkzkt_inv; fast_inverse(zkzkt, zkzkt_inv);

	A = (1.0 - eta) * A + eta * zpzct.transpose() * zpzpt_inv;
	C = (1.0 - eta) * C + eta * xkzkt * zkzkt_inv;
	//sigmaW = (1.0 - eta) * sigmaW + eta * (zczct - A * zpzct - zpzct.transpose() * A.transpose() + A * zpzpt * A.transpose()); // the latter half is redundant. Substitute A_new by its definition. However, this redundant expression stabilize the positivity of sigmas
	//sigmaV = (1.0 - eta) * sigmaV + eta * (xkxkt - C * xkzkt.transpose() - xkzkt * C.transpose() + C * zkzkt * C.transpose());  //the same as above
	sigmaW = zczct - A * zpzct - zpzct.transpose() * A.transpose() + A * zpzpt * A.transpose();
	sigmaV = xkxkt - C * xkzkt.transpose() - xkzkt * C.transpose() + C * zkzkt * C.transpose();

	//symmetrize
	symmetrize(sigmaW); symmetrize(sigmaV); symmetrize(rootVar);
}

void ForestLDS::update(  //smoothing MUST be called IN ADVANCE
	Eigen::MatrixXd& A,
	Eigen::MatrixXd& C,
	Eigen::MatrixXd& sigmaW,
	Eigen::VectorXd& meanV,
	Eigen::MatrixXd& sigmaV,
	Eigen::VectorXd& argRootMean,
	Eigen::MatrixXd& argRootVar
) {
	const double eta = 1.0; //learning rate

							//summation
	execSummation();


	//update
	argRootMean = rootMean;
	argRootVar = rootVar;

	Mat zpzpt_inv;  fast_inverse(zpzpt, zpzpt_inv);
	Mat zkzkt_inv; fast_inverse(zkzkt, zkzkt_inv);

	A = (1.0 - eta) * A + eta * zpzct.transpose() * zpzpt_inv;
	C = (1.0 - eta) * C + eta * xkzkt * zkzkt_inv;
	//sigmaW = (1.0 - eta) * sigmaW + eta * (zczct - A * zpzct - zpzct.transpose() * A.transpose() + A * zpzpt * A.transpose()); // the latter half is redundant. Substitute A_new by its definition. However, this redundant expression stabilize the positivity of sigmas
	//sigmaV = (1.0 - eta) * sigmaV + eta * (xkxkt - C * xkzkt.transpose() - xkzkt * C.transpose() + C * zkzkt * C.transpose());  //the same as above
	sigmaW = zczct - A * zpzct - zpzct.transpose() * A.transpose() + A * zpzpt * A.transpose();

	meanV = xk - C * zk;
	Eigen::MatrixXd difdift = xkxkt - C * xkzkt.transpose() - xkzkt * C.transpose() + C * zkzkt * C.transpose();
	sigmaV = difdift - meanV * meanV.transpose();

	if (m == 1 && false) {  //sigmaW becomes diagonal matrix
		//svd i.e. diagonalization by eigenvalues
		JacobiSVD<MatrixXd> svd(sigmaW, ComputeFullU | ComputeFullV);
		Mat U = svd.matrixU();
		Mat D = svd.singularValues().asDiagonal();

		//normalize C
		Mat Cnew = C * U;
		Mat D2 = Eigen::MatrixXd::Zero(n, n);
		Mat D2inv = Eigen::MatrixXd::Zero(n, n);
		for (int i = 0; i != n; i++) {
			D2(i, i) = Cnew(0, i);
			D2inv(i, i) = 1.0 / Cnew(0, i);
			C(0, i) = 1.0;
		}
		Mat L = D2 * U.transpose();
		Mat Linv = U * D2inv;

		//transformation
		A = L * A * Linv;
		sigmaW = D2 * D * D2;
		rootMean = L * rootMean;
		rootVar = L * rootVar * L.transpose();
	}
	else {  //keep sigmaW
		//svd i.e. diagonalization by eigenvalues
		//JacobiSVD<MatrixXd> svd(sigmaW, ComputeFullU | ComputeFullV);
		//Mat U = svd.matrixU();
		//Mat D = svd.singularValues().asDiagonal();

		//normalize C
		Mat Cnew = C;
		Mat D2 = Eigen::MatrixXd::Zero(n, n);
		Mat D2inv = Eigen::MatrixXd::Zero(n, n);
		for (int i = 0; i != n; i++) {
			D2(i, i) = Cnew(0, i);
			D2inv(i, i) = 1.0 / Cnew(0, i);
			C(0, i) = 1.0;
		}
		Mat L = D2;
		Mat Linv = D2inv;

		//transformation
		A = L * A * Linv;
		sigmaW = L * sigmaW * L.transpose();
		rootMean = L * rootMean;
		rootVar = L * rootVar * L.transpose();
	}


	//symmetrize
	symmetrize(sigmaW); symmetrize(sigmaV); symmetrize(rootVar);
}

void TreeLDS::checkEstimatedResult() {
	assert(leaves.size() >= 1);
	assert(m == 1);
	NodeLDS& current = o_tree[leaves[leaves.size() - 1]];

	std::ofstream out(defaultOutputFile + "check.txt");

	while (true) {
		
		out << current.observed[0] << " " << current.predObserved[0] << " " << current.Cfilter[0] << " " << current.forcasted[0] << std::endl;


		if ((excludeRoot && current.parent() == 0) || (!excludeRoot && current.parent() == -1))
			break;
		else
			current = o_tree[current.parent()];
	}
}

void TreeLDS::samplePath(const Eigen::MatrixXd& C, const Eigen::MatrixXd& sigmaV) {
	
	std::queue<int> q; //for recursion

	if (excludeRoot) {
		for (auto c : root().children) {
			NodeLDS& current = o_tree[c];
			current.sz = normal_random_variable(current.smoothe_mean, current.smoothe_var)();
			current.st = normal_random_variable(C * current.sz, sigmaV)()[0];
			for (auto cc : current.children)
				q.push(cc);
		}
	}
	else {
		root().sz = normal_random_variable(root().smoothe_mean, root().smoothe_var)();
		root().st = normal_random_variable(C * root().sz, sigmaV)()[0];
		for (auto c : root().children)
			q.push(c);
	}

	while (!q.empty()) {
		NodeLDS& current = o_tree[q.front()];
		q.pop();
		current.extendSmoothePath(C, sigmaV);

		if(!current.isLeaf())
			for (auto cc : current.children)
				q.push(cc);
	}
}

void NodeLDS::extendSmoothePath(const Eigen::MatrixXd& C, const Eigen::MatrixXd& sigmaV) {

	NodeLDS& parentNode = (*o_tree)[parent()];
	Mat Spc = parentNode.smoothe_var * J.transpose();

	//Mat Lcc = (smoothe_var - Spc.transpose() * parentNode.smoothe_var.inverse() * Spc).inverse();
	//Mat Lcp = -Lcc * Spc.transpose() * parentNode.smoothe_var.inverse();

	Vec mean = smoothe_mean + Spc.transpose() * parentNode.smoothe_var.inverse() * (parentNode.sz - parentNode.smoothe_mean);
	Mat covar = smoothe_var - Spc.transpose() * parentNode.smoothe_var.inverse() * Spc;

	sz = normal_random_variable(mean, covar)();
	st = normal_random_variable(C * sz, sigmaV)()[0];
}

void TreeLDS::outputSamplePath(std::ofstream& out) {
	for (NodeLDS& n : o_tree) {
		bool HasParent = (excludeRoot && n.parent() > 0) || (!excludeRoot && !n.isroot());
		if (HasParent)
			out << exp(o_tree[n.parent()].st) << " " << exp(n.st) << std::endl;
	}
}


void NodeLDS::outputForPaper(std::ofstream& out) {
	
	//string Rep
	out << stringRep + ", ";

	//parent string rep
	if (stringRep.size() != 1) {
		for (int i = 0; i != stringRep.size() - 1; i++)
			out << stringRep[i];
		out << ", ";
	}
	else {
		out << "-1, ";
	}

	//isLeaf
	out << isLeaf() << ", ";

	//time
	out << bornTime << ", " << bornTime + exp(observed[0]) <<", ";

	//states
	for (int i = 0; i != n; i++)
		out << smoothe_mean[i] << ", ";

	//size
	out << firstSize() << ", " << lastSize()<< ", ";

	//gfp-expression
	out << firstGFP() << ", " << lastGFP() << ", " << meanGFP();

	out << std::endl;
}


std::vector<double> TreeLDS::gentimes() {
	std::vector<double> res;
	for (NodeLDS v : o_tree)
		res.push_back(v.observed[0]);
	return res;
}

void TreeLDS::swapGentime(const std::vector<double>& gentimes, const std::vector<int>& itr) {
	assert(itr.size() == size());

	for (int i = 0; i != size(); i++) {
		o_tree[i].observed[0] = gentimes[itr[i]];
	}
}