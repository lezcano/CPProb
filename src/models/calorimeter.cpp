#include "models/calorimeter.hpp"

#include <vector>
#include <boost/random/uniform_real.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <algorithm>

#include <Rivet/Math/Matrix3.hh>
#include <Rivet/Tools/ParticleIdUtils.hh>

#include "cpprob/cpprob.hpp"
#include "cpprob/distributions/multivariate_normal.hpp"
#include "cpprob/serialization.hpp"

double min_energy_deposit(){
    return 0.005; // in GeV
}

int get_index(std::vector<double> edges, float value){
    auto index = std::distance(edges.begin(), std::find_if(edges.begin(), edges.end(),[value](double x){return x > value;})) - 1 ;
    if(0 <= index  && index < edges.size()-1){
        return index;
    }
    return -1; //out of range
}

std::vector<int> get_indices(const std::vector<double>& spacepoint, const std::vector<std::vector<double> >& edges){
    return {
            get_index(edges[0],spacepoint[0]),
            get_index(edges[1],spacepoint[1]),
            get_index(edges[2],spacepoint[2])
    };
}

std::vector<double> bin_edges(double min, double max, double nbins){
    auto binwidth = (max-min)/nbins;
    std::vector<double> bins;
    auto val = min;
    bins.push_back(val);
    for(int i = 0; i < nbins;++i){
        bins.push_back(val+=binwidth);
    }
    return bins;
}

std::vector<double> get_center(double theta, double phi, double z_begin, double shower_depth){
    double r_surface = z_begin / cos(theta); // chooses r such that vector points to plane at z = Z_BEGIN

    double offset   = shower_depth / 2.0;
    double r_factor = 1. + offset*cos(theta)/z_begin;

    //std::cout << "center is at" << r_surface*r_factor << std::endl;

    return {0,0,r_surface*r_factor};
}

std::vector<double> orient(const std::vector<double>& v, double theta, double phi){
    Rivet::Matrix3 rot1(Rivet::Vector3(0.,1.,0.), theta);
    Rivet::Matrix3 rot2(Rivet::Vector3(0.,0.,1.), phi);
    auto vp = rot2*(rot1*Rivet::Vector3(v[0],v[1],v[2]));
    return {vp[0], vp[1], vp[2]};
}

std::pair<double,std::vector<double> > shower_parameters(int pdg_id){
    if(Rivet::PID::isElectron(pdg_id) or Rivet::PID::isPhoton(pdg_id)){
        // std::cout << "EM-type shower parameters" << std::endl;
        double sampling_fraction = 0.5;
        std::vector<double> shapepars{0.2,0.2,0.50};
        return std::make_pair(sampling_fraction, shapepars);
    }
    if(Rivet::PID::isHadron(pdg_id)){
        // std::cout << "HAD-type shower parameters" << std::endl;
        double sampling_fraction = 0.25;
        std::vector<double> shapepars{0.2,0.2,1.00};
        return std::make_pair(sampling_fraction, shapepars);
    }
    return std::make_pair(0.0,std::vector<double>{0.,0.,0.});
}


std::vector<std::vector<double> > sample_particle(int pdg_id, double energy, double theta, double phi, double z_begin){

    double E_DEPOSIT = min_energy_deposit();

    auto sampling_and_shape = shower_parameters(pdg_id);
    double sampling_fraction = sampling_and_shape.first;
    std::vector<double> widths = sampling_and_shape.second;

    double mean_interactions =  energy * sampling_fraction / E_DEPOSIT;


    boost::random::poisson_distribution<int> pois(mean_interactions);
    int nsamples = cpprob::sample(pois);

    double shower_depth = 2.*5.*widths[2]; //5 sigma in z direction, times 2 for full length

    std::vector<double> center = get_center(theta, phi, z_begin, shower_depth);
    cpprob::multivariate_normal_distribution<double> multi(center, widths);

    std::vector<std::vector<double> > distribution;
    for(int i = 0; i < nsamples; ++i){
        auto s = cpprob::sample(multi);
        auto o = orient(s.values(),theta,phi);
        distribution.push_back(o);
    }

    return distribution;
}

double particle_calorimeter_response(
        int pdg_id, double energy, double theta, double phi,
        std::vector<std::vector<double> >& calo_segmentation,
        std::vector<std::vector<std::vector<double > > >& calorimeter_histo,
        const double Z_BEGIN){
    // std::cout << vectors
    using namespace cpprob;

    double total_edep = 0.0;
    // std::cout << "got a visible particle: " << p.theta() << "," << p.phi(Rivet::PhiMapping::MINUSPI_PLUSPI) << " | " << p << std::endl;

    auto samples = sample_particle(pdg_id, energy, theta, phi,Z_BEGIN);
    for(auto s : samples){
        // std::cout << "SHERPAPROBPROG SPACEPOINT" << s[0] << "," << s[1] << "," << s[2] << std::endl;
        auto idx = get_indices(s,calo_segmentation);
        if(idx[0] < 0 || idx[1] < 0 || idx[2] < 0){
            // std::cout << "skip spacepoint: " << s << std::endl;
            // std::cout << "indices: " << idx << std::endl;
            continue;
        }
        float edep = min_energy_deposit();
        calorimeter_histo[idx[0]][idx[1]][idx[2]] += edep;
        total_edep += edep;
    }
    // std::cout << "particle deposited a " << total_edep / p.E() *100 << " percent of its energy into the calorimeter" << std::endl;
    return total_edep;
}

std::vector<std::vector<std::vector<double > > > calo_simulation(const std::vector<std::vector<double> >& particle_data){

    // std::cout << "simulating calo response of " << particle_data.size() << " particles" << std::endl;

    int NBINX = 35;
    int NBINY = 35;
    int NBINZ = 20;
    double Z_BEGIN = 4;

    std::vector<std::vector<double> > histo_edges{bin_edges(-3,3,NBINX),bin_edges(-3,3,NBINY),bin_edges(Z_BEGIN,15,NBINZ)};
    std::vector<std::vector<std::vector<double > > > histo(
            NBINX, std::vector<std::vector<double> >(
                    NBINY, std::vector<double>(
                            NBINZ,0.0))
    );

    double total_edep = 0;

    for(auto particle_data : particle_data){
        // std::cout << "SHERPAPROBPROG BEGIN PARTICLE " << p << std::endl;

        bool calo_visible = particle_data[7];
        int pdg_id = particle_data[6];
        double energy = particle_data[3];
        double theta = particle_data[4];
        double phi = particle_data[5];

        // std::cout << "pdg id: " << pdg_id << std::endl;

        if(pdg_id == -99999) continue;
        if(!calo_visible){
          // std::cout << "invisible particle" << std::endl;
          continue;
        }

        // std::cout << "shower: " << energy << std::endl;
        double particle_edep = particle_calorimeter_response(pdg_id, energy, theta, phi, histo_edges,histo,Z_BEGIN);
        total_edep += particle_edep;
        // std::cout << "SHERPAPROBPROG END PARTICLE " << p << std::endl;
    }
    // std::cout << "deposited total of " << total_edep << std::endl;

    return histo;
}
