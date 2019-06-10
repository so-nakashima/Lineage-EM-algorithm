#include "Tree.h"
#include "test.h"
#include "dataGenerator.h"
#include "estimator.h"
#include "execution-estimator.h"

#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <fstream>

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/polygamma.hpp>


void treeDescription();

int main(){
	//estimateTree2();
	//generateTree2forest();

	//tree2Stat();

	//tree2Estimator();
	//tree1Estimator();

	//generateTrees();
	//WakamotoEstimator("toy", ".\\paper\\res\\Wakamoto_type2.txt", 2);

	//treeDescription();

	
	//LDSinferenceTest();
	//LDSestimateTest();
	//carpetBombingTest();


	//for (int i = 1; i != 11; i++) {
	//	std::string filename = ".\\LDS1\\F3_LVS4_Glc_37C_1min_meanV_nonDiagonalizationOfSigmaW_n=" + std::to_string(i) + ".txt";
	//	WakamotoEstimatorLDSwithMeanV("F3_LVS4_Glc_37C_1min", filename, i);
	//}
	//for (int i = 1; i != 11; i++) {
	//	std::string filename = ".\\LDS1\\F3_rpsL_Glc_37C_1min_fixedMeanV_nonDiagonalizationOfSigmaW_n=" + std::to_string(i) + ".txt";
	//	WakamotoEstimatorLDSwithMeanV("F3_rpsL_Glc_37C_1min", filename, i);
	//}

	//CheckEstimatedResult("F3_rpsL_Glc_37C_1min");
	//CheckEstimatedResult("F3_LVS4_Glc_37C_1min");
	//checkDistributionsSteady();
	//CheckSmoothedPath("Br_Gly_37C_1min");
	//CheckSmoothedPath("Br_Glc_37C_1min");
	//CheckSmoothedPath("F3_rpsL_Glc_37C_1min");
	//CheckSmoothedPath("F3_LVS4_Glc_37C_1min");

	/*generateLDSForest();
	for(int i = 1; i != 11; i++)
		carpetBombingForestTestForGeneratedData(i, 1);*/

	 //AICselctions(100, 6);

	//std::string filename = ".\\LDS1\\F3_rpsL_Glc_37C_1min_nonDiagonalizationOfSigmaW_nonMeanV_bootstrap_n=";
	//WakamotoEstimatorLDS_Bootstrap("F3_rpsL_Glc_37C_1min", filename);

	AICselctionsBootStrap(100, "F3_rpsL_Glc_37C_1min", 5);

	return 0;
}

void treeDescription() {
	std::string filename = ".\\Tree2_220min\\Tree2_0.dat";
	Tree t;
	t.load(filename);
	t.MathematicaGraphics("EMBL2");
}