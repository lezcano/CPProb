//
// Created by lezkus on 7/05/17.
//

#include <iostream>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <cpprob/distributions/vmf.hpp>
#include <cpprob/distributions/multivariate_normal.hpp>
#include <cpprob/cpprob.hpp>
#include <math.h>

int select(){
    boost::random::uniform_smallint<int> discrete {0, 2};
    int select = cpprob::sample(discrete);

    cpprob::predict(select);

    switch(select){
        case 0:
            return 10;
        case 1:
            return 20;
        case 2:
            return 30;
    }
}

int main(){
    auto moms = select();
}
