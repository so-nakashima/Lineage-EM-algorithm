#include "Tree.h"
#include "assert.h"
#include <fstream>
#include <iostream>
#include <queue>


Node::Node()
{
}

Node::Node(double survivalTime, int typeno, int parentsID,std::vector<Node>* tree)
	: survivalTime(survivalTime), o_isLeaf(false), o_tree(tree), o_parent(parentsID){
	currentDistribution = std::vector<double>(typeno, 0);
	alpha = std::vector<double>(typeno, 0);
	beta = std::vector<double>(typeno, 0);

	if (parentsID >= 0) {
		o_borntime = (*tree)[parentsID].bornTime() + (*tree)[parentsID].survivalTime;
		if ((*tree)[parentsID].left() == -1)
			o_stringrep = (*tree)[parentsID].stringRep() + "1";
		else
			o_stringrep = (*tree)[parentsID].stringRep() + "2";
	}
	else {
		o_borntime = 0;
		o_stringrep = "1";
	}
}

Node::Node(int type, double survivalTime,int typeno, int parentsID, std::vector<Node>* tree) 
	: survivalTime(survivalTime), o_isLeaf(true), o_tree(tree), o_parent(parentsID){
	if (type != -1) {
		o_type = type;
		currentDistribution = std::vector<double>(typeno, 0);
		currentDistribution[type] = 1.0;
		alpha = std::vector<double>(typeno, 0);
		beta = std::vector<double>(typeno, 0);
		beta[type] = 1;

		if (parentsID >= 0) {
			o_borntime = (*tree)[parentsID].bornTime() + (*tree)[parentsID].survivalTime;
			if ((*tree)[parentsID].left() == -1)
				o_stringrep = (*tree)[parentsID].stringRep() + "1";
			else
				o_stringrep = (*tree)[parentsID].stringRep() + "2";
		}
		else {
			o_borntime = 0;
			o_stringrep = "1";
		}
	}
	else {  //this is leaf, but its type is unknown
		currentDistribution = std::vector<double>(typeno, 0);
		alpha = std::vector<double>(typeno, 0);
		beta = std::vector<double>(typeno, 0);
		for (int i = 0; i != typeno; i++)
			beta[i] = 1;

		if (parentsID >= 0) {
			o_borntime = (*tree)[parentsID].bornTime() + (*tree)[parentsID].survivalTime;
			if ((*tree)[parentsID].left() == -1)
				o_stringrep = (*tree)[parentsID].stringRep() + "1";
			else
				o_stringrep = (*tree)[parentsID].stringRep() + "2";
		}
		else {
			o_borntime = 0;
			o_stringrep = "1";
		}
	}
}	

bool Node::isleft() const {
	if (isroot())
		return false;
	Node parentNode = (*o_tree)[parent()];
	return this == &((*o_tree)[parentNode.left()]);
}

Node::~Node()
{
}

Tree::Tree()
{
}

Tree::~Tree()
{
}

bool Node::addChild(int childsId) {
	assert(!isLeaf());
	if (left() == -1) {
		o_left = childsId;
		return true;
	}
	else if (right() == -1) {
		o_right = childsId;
		return true;
	}
	else
		return false;
}

void Tree::load(std::string filename) {
	std::ifstream file(filename.c_str());
	if (!file) {
		std::cout << "Cannot open \"" << filename << "\"." << std::endl;
		return;
	}

	int N, X; //number of nodes, types
	file >> N >> X;
	typeno = X;

	//load nodes
	o_tree.clear();
	for (int i = 0; i != N; i++) {
		int isleaf, parent; file >> parent >> isleaf;
		if (isleaf) {
			int type; double survivalTime;
			file >> type >> survivalTime;
			o_tree.push_back(Node(type, survivalTime, X, parent,&o_tree));
			leaves.push_back(i);
			
			//increment the cardinality of leaves
			leavesNo++;
		}
		else {
			double survivaltime; file >> survivaltime;
			o_tree.push_back(Node(survivaltime, X, parent,&o_tree));
		}
		if (i != 0)
			o_tree[parent].addChild(i);
	}
}

void Tree::load(int typeNo, std::string filename, bool paperFlag) {  //For Wakamoto Data, i.e.  type = -1 for EVERY entry (even for bulk entry), the last argument is used only for paper
	std::ifstream file(filename.c_str());
	if (!file) {
		std::cout << "Cannot open \"" << filename << "\"." << std::endl;
		return;
	}

	typeno = typeNo;

	//for paper
	if (paperFlag) {
		o_tree.clear();
		int parent;
		int count = 0;
		file >> parent >> parent; //trash irrelevant part
		while (file >> parent) {
			int isleaf; file >> isleaf;
			if (isleaf) {
				int type; double survivalTime;
				file >> type >> survivalTime;
				o_tree.push_back(Node(-1, survivalTime, typeno, parent, &o_tree));
				leaves.push_back(count);

				//increment the cardinality of leaves
				leavesNo++;
			}
			else {
				double survivaltime;
				file >>  survivaltime;
				o_tree.push_back(Node(survivaltime, typeno, parent, &o_tree));
			}
			if (count != 0)
				o_tree[parent].addChild(count);

			count++;
		}

		return;
	}

	//load nodes
	o_tree.clear();
	int parent;
	int count = 0;
	while(file >> parent){
		int isleaf; file >> isleaf;
		if (isleaf) {
			int type; double survivalTime;
			file >> type >> survivalTime;
			o_tree.push_back(Node(-1, survivalTime, typeno, parent, &o_tree));
			leaves.push_back(count);

			//increment the cardinality of leaves
			leavesNo++;
		}
		else {
			int type;
			double survivaltime; 
			file >> type >> survivaltime;
			o_tree.push_back(Node(survivaltime, typeno, parent, &o_tree));
		}
		if (count != 0)
			o_tree[parent].addChild(count);

		count++;
	}
}


void Tree::inference(const std::vector<std::vector<double>>& transitMat, const std::vector<Density*>& densities) {
	//update alpha and beta
	calcBeta(transitMat, densities);
	calcAlpha(transitMat, densities);
	//update currentDistricution
	for (int i = 0; i != o_tree.size(); i++) {
		for (int j = 0; j != typeno; j++) {
			o_tree[i].currentDistribution[j] = o_tree[i].alpha[j] * o_tree[i].beta[j];
		}
		//normalization
		normalize(o_tree[i].currentDistribution);
	}
}

void Tree::calcAlpha(const std::vector<std::vector<double>>& transitMat, const std::vector<Density*>& densities) {  //beta MUST be updated in advance
	//assertions
	assert(steadyDistribution.size() == typeno);
	assert(densities.size() == typeno);
	assert(transitMat.size() == typeno);

	//initialize root node
	for (int i = 0; i != typeno; i++) {
		assert(densities[i] != NULL);
		root().alpha[i] = (1.0 - densities[i]->CDF(root().survivalTime)) * steadyDistribution[i];
	}

	//recursive computation
	std::queue<int> q;
	q.push(root().left());
	q.push(root().right());
	while (!q.empty()) {
		Node& current = o_tree[q.front()];
		q.pop();
		current.updateAlpha(transitMat, densities);

		//for children
		if (!current.isLeaf()) {
			q.push(current.left());
			q.push(current.right());
		}
	}
}

void Tree::calcBeta(const std::vector<std::vector<double>>& transitMat, const std::vector<Density*>& densities) {
	//assertions
	assert(steadyDistribution.size() == typeno);
	assert(densities.size() == typeno);
	assert(transitMat.size() == typeno);

	////initialization (necessary for Wakamoto-type data)
	//for (int i = 0; i != leaves.size(); i++) {
	//	Node& current = o_tree[leaves[i]];

	//	if (current.type() != -1)  //Done nothing for type-known nodes
	//		current.changeC(1.0 - densities[current.type()]->CDF(current.survivalTime));
	//	else {  //For type-unkonwn leaves
	//		std::vector<double> betacomp(typeno, 0);
	//		for (int j = 0; j != typeno; j++) {
	//			betacomp[j] = 1.0 - densities[j]->CDF(current.survivalTime);
	//		}
	//		double c = normalize(betacomp);
	//		current.beta = betacomp;
	//		current.changeC(c);
	//	}
	//}


	//recursive computation
	std::queue<int> q;
	for (int i = 0; i != leaves.size(); i++) {
		Node current = o_tree[leaves[i]];
		//if (current.isLeaf())  //why? redundant condition?
			q.push(current.parent());
	}
	while (!q.empty()) {
		Node& current = o_tree[q.front()];
		q.pop();
		//update
		bool flag = current.updateBeta(transitMat, densities);
		//recursion if update sucessed 
		if (flag && !current.isroot())
			q.push(current.parent());
	}
}

void Node::updateAlpha(const std::vector<std::vector<double>>& transitMat, const std::vector<Density*>& densities) { 
	Node parentNode = o_tree->at(parent());
	Node sisterNode = o_tree->at(sister());
	for (int i = 0; i != o_typeno(); i++) {
		//calculate summation part first
		double sum = 0;
		for (int j = 0; j != o_typeno(); j++) {
			assert(transitMat[j].size() == o_typeno());

			double subsum = 0.0;  //factors concerning sister node
			for (int k = 0; k != o_typeno(); k++) {
				if (sisterNode.isLeaf())
					subsum += transitMat[j][k] * (1.0 - densities[k]->CDF(sisterNode.survivalTime));
				else
					subsum += transitMat[j][k] * densities[k]->density(sisterNode.survivalTime);
			}

			sum += parentNode.alpha[j] * transitMat[j][i] * subsum;
		}
		//then get the result
		if(isLeaf())
			alpha[i] = sum * (1.0 - densities[i]->CDF(survivalTime));  //see note on 30/6/2017
		else
			alpha[i] = sum * densities[i]->density(survivalTime);
	}

	//normalization
	normalize(alpha);
}

bool Node::updateBeta(const std::vector<std::vector<double>>& transitMat, const std::vector<Density*>& densities) {
	//update start if two children are updated
	betaUpdateFlag = !betaUpdateFlag;
	if (!betaUpdateFlag)
		return false;

	Node leftNode = o_tree->at(left());
	Node rightNode = o_tree->at(right());

	for (int i = 0; i != o_typeno(); i++) {
		//beta = leftsum * rightsum.
		//first compute leftsum
		double leftsum = 0;
		for (int j = 0; j != o_typeno(); j++) {
			if(leftNode.isLeaf())
				leftsum += leftNode.beta[j] * (1.0 - densities[j]->CDF(leftNode.survivalTime)) * transitMat[i][j];
			else
				leftsum += leftNode.beta[j] * densities[j]->density(leftNode.survivalTime) * transitMat[i][j];
		}
		//then compute rightsum
		double rightsum = 0;
		for (int j = 0; j != o_typeno(); j++) {
			if (rightNode.isLeaf())
				rightsum += rightNode.beta[j] * (1.0 - densities[j]->CDF(rightNode.survivalTime)) * transitMat[i][j];
			else
				rightsum += rightNode.beta[j] * densities[j]->density(rightNode.survivalTime) * transitMat[i][j];
		}
		beta[i] = leftsum * rightsum;
	}

	//normalization
	o_c = normalize(beta);

	//notify that update suceeded
	return true;
}

void Tree::steadyState(double experimentTime) {
	std::vector<int> count(typeno, 0);// count # ofleaves of each type 

	for (auto itr = leaves.begin(); itr != leaves.end(); itr++) {
		count[o_tree[*itr].type()]++;
	}

	//output
	std::cout << "steady distribution is: ";
	for (int i = 0; i != typeno; i++) {
		std::cout << ((double)count[i]) / leaves.size() << " ";
		if (i == typeno - 1)
			std::cout << std::endl;
	}
	std::cout << "growth rate is: " << log(leaves.size()) / experimentTime << std::endl;
}

std::vector<double> Tree::sum(std::function<double(const Node&, int)> f) {
	std::vector<double> res(typeno, 0.0);
	std::queue<int> q; //queue which stores id, for recursive computation
	q.push(0);
	while (!q.empty()) {
		Node current = o_tree[q.front()];
		q.pop();
		for (int i = 0; i != typeno; i++) {
			res[i] += f(current, i);
		}
		if (!current.isLeaf()) {
			q.push(current.left());
q.push(current.right());
		}
	}
	return res;
}

std::vector<std::vector<double>> Tree::transitCoef(const std::vector<std::vector<double>>& transit, const std::vector<Density*>& densities) {
	//hyper-parameter. if P[x_i = i] < epsilon, then skip to avoid 0 division in normalization
	const double epsilon = 1.0e-5;

	std::vector<std::vector<double>> res(typeno, std::vector<double>(typeno, 0));
	std::queue<int> q; //queue which stores id, for recursive computation
	q.push(0);
	while (!q.empty()) {
		Node& current = o_tree[q.front()];
		q.pop();
		if (current.isLeaf()) //skip leaf node (since no contribution)
			continue;
		else {
			//sum
			for (int i = 0; i != typeno; i++) { //type of current
				//skip if  P[x_i = i] is too small
				if (current.currentDistribution[i] < epsilon)
					continue;

				// to store the result for this-left(L), this-right(R), normalized after.
				std::vector<std::vector<double>> temp(typeno, std::vector<double>(typeno, 0));
				double sumL, sumR;
				sumL = sumR = 0;
				std::vector<double> tempL, tempR;
				tempL = tempR = std::vector<double>(typeno, 0);

				//old(wrong ver. ?)
				//non-normalized computation
				for (int j = 0; j != typeno; j++) { //type of children
					Node& left = o_tree[current.left()];
					Node& right = o_tree[current.right()];

					//for left child 
					if (left.isLeaf())  //if left is leaf, then use 1-CDF instead of density, see note on 30/6/2017;
						tempL[j] += transit[i][j] * left.beta[j] * (1.0 - densities[j]->CDF(left.survivalTime)); //for details, see not on the 19 June 2017
					else
						tempL[j] += transit[i][j] * left.beta[j] * densities[j]->density(left.survivalTime);

					sumL += tempL[j];

					//for right child
					if (right.isLeaf())
						tempR[j] += transit[i][j] * current.alpha[i] * right.beta[j] * (1.0 - densities[j]->CDF(right.survivalTime));
					else
						tempR[j] += transit[i][j] * current.alpha[i] * right.beta[j] * densities[j]->density(right.survivalTime);
					sumR += tempR[j];
				}

				//normalize and add to res
				for (int j = 0; j != typeno; j++) {
					res[i][j] += (tempL[j] / sumL + tempR[j] / sumR) * current.currentDistribution[i];
				}

				//non-normalized computation
				//for (int j = 0; j != typeno; j++) { //type of left children
				//	for (int k = 0; k != typeno; k++) { //type of right children
				//		Node& left = o_tree[current.left()];
				//		Node& right = o_tree[current.right()];


				//		double tempL, tempR;

				//		//for left child 
				//		if (left.isLeaf())  //if left is leaf, then use 1-CDF instead of density, see note on 30/6/2017;
				//			tempL = transit[i][j] *  left.beta[j] * (1.0 - densities[j]->CDF(left.survivalTime)); //for details, see not on the 19 June 2017
				//		else
				//			tempL = transit[i][j] *  left.beta[j] * densities[j]->density(left.survivalTime);

				//		//for right child
				//		if (right.isLeaf())
				//			tempR = transit[i][k]  * right.beta[k] * (1.0 - densities[k]->CDF(right.survivalTime));
				//		else
				//			tempR = transit[i][k]  * right.beta[k] * densities[k]->density(right.survivalTime);
				//			
				//		temp[j][k] = tempL * tempR * current.alpha[i];
				//		sum += temp[j][k];
				//	}
				//}

				////normalize and add to res
				//for (int j = 0; j != typeno; j++) {
				//	for (int k = 0; k != typeno; k++) {
				//		res[i][j] += temp[j][k] / sum;
				//		res[i][k] += temp[j][k] / sum;
				//	}
				//}

				/* //non-normalized ver.
				for (int j = 0; j != typeno; j++) { //type of children
					Node& left = o_tree[current.left()];
					Node& right = o_tree[current.right()];

					//for left child
					res[i][j] += transit[i][j] * current.alpha[i] * left.beta[j] * densities[j]->density(left.survivalTime); //for details, see not on the 19 June 2017
					//for right child
					res[i][j] += transit[i][j] * current.alpha[i] * right.beta[j] * densities[j]->density(right.survivalTime);
				}*/

			}


			//recursion
			q.push(current.left());
			q.push(current.right());
		}
	}
	return res;
}

double normalize(std::vector<double>& vec) {
	//compute sum
	double sum = 0.0;
	for (int i = 0; i != vec.size(); i++)
		sum += vec[i];

	//normalize
	for (int i = 0; i != vec.size(); i++)
		vec[i] /= sum;
	return sum;
}

double Tree::logLik(bool leafIncluded) { //see my not on 22/9/2017
	double res = 0;

	//root node
	double sum = 0;
	for (int i = 0; i != typeno; i++) {
		sum += root().alpha[i] * root().beta[i];
	}
	res += log(sum);

	//rest of nodes
	for (int i = 1; i < o_tree.size(); i++)
		if(leafIncluded || !(o_tree[i].isLeaf()))
			res += log(o_tree[i].c());

	return res;
}

int Node::sister() const {
	if (isroot())  //except root case
		return -1;

	int parentID = parent();
	int leftID = (*o_tree)[parentID].left();
	int rightID = (*o_tree)[parentID].right();
	if (isleft())
		return rightID;
	else
		return leftID;
}

void Tree::MathematicaGraphics(std::string outPath) {
	std::ofstream out(outPath);
	//out << "lt = {";
	//for (int i = 0; i != o_tree.size(); i++) {
	//	out << "{";

	//	//index
	//	out << "{";
	//	for (int j = 0; j != o_tree[i].stringRep().size(); j++) {
	//		out << o_tree[i].stringRep()[j];
	//		if (j != o_tree[i].stringRep().size() - 1)
	//			out << ", ";
	//	}
	//	out << "}, ";

	//	//status
	//	out << "{";
	//	out <<  1 << ", " << o_tree[i].survivalTime << ", " << o_tree[i].bornTime() << ", " << o_tree[i].bornTime() + o_tree[i].survivalTime;
	//	out << "}, ";

	//	//death or not
	//	if (o_tree[i].isLeaf())
	//		out << "True, ";
	//	else
	//		out << "False, ";

	//	//parent status
	//	int parentID = o_tree[i].parent();
	//	out << "{";
	//	if (i != 0) {
	//		out <<  1 << ", " << o_tree[parentID].survivalTime << ", " << o_tree[parentID].bornTime() << ", " << o_tree[parentID].bornTime() + o_tree[parentID].survivalTime;
	//	}
	//	else
	//		out << "1, 0, 0, 0";
	//	out << "}";

	//	if (i != o_tree.size() - 1)
	//		out << "}, ";
	//	else
	//		out << "}";
	//}


	//out << "}" << std::endl;

	//for paper
	for (int i = 0; i != o_tree.size(); i++) {

		//index
		out << o_tree[i].stringRep();
		out << ", ";


		//parent-index
		for (int j = 0; j != o_tree[i].stringRep().size() - 1; j++) {
			out << o_tree[i].stringRep()[j];
		}
		out << ", ";

		//death or not
		if (o_tree[i].isLeaf())
			out << "1, ";
		else
			out << "0, ";

		//time
		out << o_tree[i].bornTime() << ", " << o_tree[i].bornTime() + o_tree[i].survivalTime << ", ";
		
		//states
		out << o_tree[i].currentDistribution[0] << ", " << o_tree[i].currentDistribution[1];

		out << std::endl;
	}

}