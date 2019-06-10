#pragma once
void integrate(std::vector<double>& res, const std::function<double(double)>& f, double dt, double min, double max);
void tree1EstimatorCore(std::vector<Tree*>& forest, std::vector<std::vector<double>>& transit, double& lambda0, double& lambda1, std::ofstream& resultFile);
void tree2EstimatorCore(std::vector<Tree*>& forest, std::vector<std::vector<double>>& transit, double& alpha0, double& alpha1, double& beta0, double& beta1, std::ofstream& resultFile, bool leafIncluded = true, bool weightCorrection = false);
void WakamotoEstimatorCore(std::vector<Tree*>& forest, std::vector<std::vector<double>>& transit, std::vector<double>& alpha, std::vector<double>& beta, double maxSurvival, std::ofstream& resultFile, bool leafIncluded = true, bool weightCorrection = false);