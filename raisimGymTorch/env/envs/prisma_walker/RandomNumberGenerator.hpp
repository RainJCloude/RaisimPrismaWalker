#ifndef RANDOMNUMBERGENERATOR_HPP_
#define RANDOMNUMBERGENERATOR_HPP_

// for random sampling
#include <random>
#include <cstdlib>
#include <mutex>
#include <Eigen/Core>


        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); 


template<typename Dtype>
class RandomNumberGenerator {

 public:

  RandomNumberGenerator() {
  }

  ~RandomNumberGenerator() {
  }

 
  /* from -1 to 1*/
  Dtype sampleUniform() {

        auto dist = std::uniform_real_distribution<Dtype>(-1, 1);
        return dist(gen);
  }

  /* from 0 to 1*/
  Dtype sampleUniform01() {

        auto dist = std::uniform_real_distribution<Dtype>(0, 1);
        return dist(gen);
  }

  int intRand(const int &min, const int &max) {

        auto dist= std::uniform_int_distribution<>(min, max);  //se lo prendeva float e mi dava problemi
        return dist(gen);
    
  }



};


#endif /* RANDOMNUMBERGENERATOR_HPP_ */