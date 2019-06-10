#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

using namespace Eigen;
typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;

int generation = 15;

//int n = 2;
//int m = 2;
//Mat A(n, n);
//A << 1, 0, 0, 1;
//Mat C(m, n);
//C << 1, 0, 0, 1;
//Mat sigmaW(n, n);
//sigmaW << .001, 0, 0, .001;
//Mat sigmaV(m, m);
//sigmaV << .1, 0, 0, .1;
//Vec initState(2);
//initState << 0, 0;

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
sigmaV << .1, 0, 0, .1;
Vec initState(2);
initState << 1, 0;