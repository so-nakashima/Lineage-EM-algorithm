#pragma once
#include <boost/math/special_functions/gamma.hpp>

class Density
{
public:
	Density() {};
	~Density() {};

	virtual double density(double x) {
		return 1.0; //i.e. no age effect
	}
	virtual double CDF(double x) { //cummulative density function
		return 0.0;
	}

private:

};

class ExponentialDensity : public Density
{
public:
	ExponentialDensity() {};
	ExponentialDensity(double lambda) : lambda(lambda) {};
	~ExponentialDensity() {};

	virtual double density(double x) { return lambda * exp(-lambda * x); }
	virtual double CDF(double x) { return 1.0 - exp(- lambda * x); }
	void setLambda(double x) { lambda = x; }
private:
	double lambda;
};

class GammaDensity :public Density
{
public:
	GammaDensity(double alpha, double beta) : alpha(alpha), beta(beta) {};

	//virtual double density(double x) { return pow(beta, alpha) * pow(x, alpha - 1) * exp(-beta * x) / boost::math::tgamma(alpha); }
	virtual double density(double x) { return  boost::math::gamma_p_derivative(alpha, x * beta) * beta; }
	virtual double CDF(double x) { return boost::math::gamma_p(alpha, x * beta); }
	void setParameter(double alphaNew, double betaNew) { alpha = alphaNew; beta = betaNew; }
private:
	double alpha, beta;
};

