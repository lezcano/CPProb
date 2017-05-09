#include "models/sherpa_mini.hpp"


#include <iostream>
#include <boost/random/normal_distribution.hpp>
#include <cpprob/distributions/vmf.hpp>
#include <cpprob/distributions/multivariate_normal.hpp>
#include <cpprob/cpprob.hpp>
#include <math.h>
#include "YODA/Histo2D.h"
#include "Rivet/Math/Vector4.hh"

namespace models {

std::pair<double, std::vector<std::vector<double> > > energy_deposits(const Rivet::FourMomentum &mom) {
	double eta = mom.eta();
	double phi = mom.phi(Rivet::PhiMapping::MINUSPI_PLUSPI);

	double WIDTH = 0.05;
	double NSAMPLES = 1000.;

	cpprob::multivariate_normal_distribution<double> multi({eta, phi}, WIDTH);

	std::vector<std::vector<double> > distribution;
	for (int i = 0; i < NSAMPLES; ++i) {
		distribution.push_back(cpprob::sample(multi));
	}

	double mini_e = mom.E() / NSAMPLES;

	return std::make_pair(mini_e, distribution);
}

std::vector<Rivet::FourMomentum> select() {
	boost::random::uniform_real_distribution<double> real_uniform{0, 4};
	double select_ran = cpprob::sample(real_uniform, true);
	int select = int(select_ran);
	cpprob::predict(select);

	std::cerr << "selected channel " << select << std::endl;

	auto v0 = Rivet::FourMomentum::mkXYZE(3.12206631, 0.18609799, -0.13257316, 3.16910447);
	auto v1 = Rivet::FourMomentum::mkXYZE(0.46751203, -0.18594433, -0.08841184, 0.52956513);
	auto v2 = Rivet::FourMomentum::mkXYZE(3.03845425, -0.35807276, 0.05962087, 3.06324252);
	auto v3 = Rivet::FourMomentum::mkXYZE(3.37196741, 0.35791909, 0.16136414, 3.39474722);

	switch (select) {
		case 0:
			return {v0};
		case 1:
			return {v0, v1};
		case 2:
			return {v0, v1, v2};
		case 3:
			return {v0, v1, v2, v3};
	}
}

std::vector<double> dummy_sherpa() {
	int NETA = 100;
	int NPHI = 100;
	YODA::Histo2D histo(NETA, -1, 1, NPHI, -1, 1);

	auto moms = select();

	for (auto mom : moms) {
		auto en = energy_deposits(mom);
		for (auto x : en.second) {
			histo.fill(x[0], x[1], en.first);
		}
	}

	std::vector<double> histovals;
	std::transform(histo.bins().begin(), histo.bins().end(), std::back_inserter(histovals),
				   [](const YODA::HistoBin2D &b) {
					   return b.sumW();
				   }
	);
	return histovals;
}


void sherpa_wrapper(const std::vector<double> &test_image) {
	std::vector<double> sherpa_img = dummy_sherpa();
	double OBS_WIDTH = 1.0;
	cpprob::multivariate_normal_distribution<double> obs_distr(sherpa_img.begin(), sherpa_img.end(), OBS_WIDTH);
	cpprob::observe(obs_distr, test_image);
}

} // end namespace models