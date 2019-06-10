#pragma once
void tree2Estimator();
void tree1Estimator();
void tree2Stat();
void tree1Stat();
void WakamotoEstimator(std::string experiment, std::string resultFile, int typeno);
void ReadWakamoto(std::string experiment, std::vector<Tree*>& forest, int typeno, double& alpha, double& beta, double& maxSurvival);