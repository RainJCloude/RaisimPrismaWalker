//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "RaisimGymEnv.hpp"
#include "omp.h"
#include "Yaml.hpp"
#include "Environment.hpp"

namespace raisim {

int THREAD_COUNT;

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg, bool normalizeObservation=true)
      : resourceDir_(resourceDir), cfgString_(cfg), normalizeObservation_(normalizeObservation) {
    Yaml::Parse(cfg_, cfg);

    if(&cfg_["render"])
      render_ = cfg_["render"].template As<bool>();
    init();
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  const std::string& getResourceDir() const { return resourceDir_; }
  const std::string& getCfgString() const { return cfgString_; }

  void init() {
    THREAD_COUNT = cfg_["num_threads"].template As<int>();
    omp_set_num_threads(THREAD_COUNT);
    num_envs_ = cfg_["num_envs"].template As<int>();

    environments_.reserve(num_envs_);
    rewardInformation_.reserve(num_envs_);
    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template As<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template As<double>());
      rewardInformation_.push_back(environments_.back()->getRewards().getStdMap());
    }

    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    actionDim_ = environments_[0]->getActionDim();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")

    /// ob scaling
    if (normalizeObservation_) {
      obMean_.setZero(obDim_);
      obVar_.setOnes(obDim_);
      recentMean_.setZero(obDim_);
      recentVar_.setZero(obDim_);
      delta_.setZero(obDim_);
      epsilon.setZero(obDim_);
      epsilon.setConstant(1e-8);
    }
  }

  // resets all environments and returns observation
  void reset() {
    for (auto env: environments_)
      env->reset();
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));

    if (normalizeObservation_)
      updateObservationStatisticsAndNormalize(ob, updateStatistics);
  }

  void select_heightMap(){
    for (auto *env: environments_)
      env->select_heightMap();
  }
  

  void select_terrain_from_tester(float stepHeight){
		environments_[0]->select_terrain_from_tester(stepHeight);
	}

  void getMotorTorques(Eigen::Ref<EigenVec> &tau){
    environments_[0]->getMotorTorques(tau);
	}

  void getpTarget(Eigen::Ref<EigenVec> &pTarget){
    environments_[0]->getReference(pTarget);
	}

  void command_vel(const double omega_vel){
    environments_[0]->command_vel(omega_vel);
	}

  void getJointPositions(Eigen::Ref<EigenVec> &q){
    environments_[0]->getJointPositions(q);
	}

  void getJointVelocities(Eigen::Ref<EigenVec> & dotq){
    environments_[0]->getJointVelocities(dotq);
	}

  void getActualTorques(Eigen::Ref<EigenVec> &tau){
    environments_[0]->getActualTorques(tau);
	}

  void getPitch(Eigen::Ref<EigenVec>& pitch){
		environments_[0]->getPitch(pitch);
	}

	void getYaw(Eigen::Ref<EigenVec>& yaw){
		environments_[0]->getYaw(yaw);
	}

	void getAngularVel(Eigen::Ref<EigenVec>& angVel){
		environments_[0]->getAngularVel(angVel);
	}
	
	void getCurrentAction(Eigen::Ref<EigenVec>& currAct){
		environments_[0]->getCurrentAction(currAct);
	}

  void getJointAccelerations(Eigen::Ref<EigenVec> &acc){
        raisim::VectorizedEnvironment<ChildEnvironment>::environments_[0]->getJointAccelerations(acc);
  }
  
  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(auto)
    for (int i = 0; i < num_envs_; i++)
      perAgentStep(i, action, reward, done);
  }

  void turnOnVisualization() { if(render_) environments_[0]->turnOnVisualization(); }
  void turnOffVisualization() { if(render_) environments_[0]->turnOffVisualization(); }
  void startRecordingVideo(const std::string& videoName) { if(render_) environments_[0]->startRecordingVideo(videoName); }
  void stopRecordingVideo() { if(render_) environments_[0]->stopRecordingVideo(); }
  void getObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float &count) {
    mean = obMean_; var = obVar_; count = obCount_; }
  void setObStatistics(Eigen::Ref<EigenVec> &mean, Eigen::Ref<EigenVec> &var, float count) {
    obMean_ = mean; obVar_ = var; obCount_ = count; }

  void setSeed(int seed) {
    int seed_inc = seed;

    #pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++)
      environments_[i]->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  int getObDim() { return obDim_; }
  int getActionDim() { return actionDim_; }
  int getNumOfEnvs() { return num_envs_; }

  ////// optional methods //////
  void curriculumUpdate() {
    for (auto *env: environments_)
      env->curriculumUpdate();
  };

  const std::vector<std::map<std::string, float>>& getRewardInfo() { return rewardInformation_; }

 private:
  void updateObservationStatisticsAndNormalize(Eigen::Ref<EigenRowMajorMat> &ob, bool updateStatistics) {
    if (updateStatistics) {
      recentMean_ = ob.colwise().mean();
      recentVar_ = (ob.rowwise() - recentMean_.transpose()).colwise().squaredNorm() / num_envs_;

      delta_ = obMean_ - recentMean_;
      for(int i=0; i<obDim_; i++)
        delta_[i] = delta_[i]*delta_[i];

      float totCount = obCount_ + num_envs_;

      obMean_ = obMean_ * (obCount_ / totCount) + recentMean_ * (num_envs_ / totCount);
      obVar_ = (obVar_ * obCount_ + recentVar_ * num_envs_ + delta_ * (obCount_ * num_envs_ / totCount)) / (totCount);
      obCount_ = totCount;
    }

#pragma omp parallel for schedule(auto)
    for(int i=0; i<num_envs_; i++)
      ob.row(i) = (ob.row(i) - obMean_.transpose()).template cwiseQuotient<>((obVar_ + epsilon).cwiseSqrt().transpose());
  }

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done) {
                                float terminalReward = 0;

    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    reward[agentId] = environments_[agentId]->step(action.row(agentId));

    rewardInformation_[agentId] = environments_[agentId]->getRewards().getStdMap();

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }
  }

  
  std::vector<std::map<std::string, float>> rewardInformation_;

  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0;
  bool recordVideo_=false, render_=false;
  std::string resourceDir_;
  Yaml::Node cfg_;
  std::string cfgString_;

  /// observation running mean
  bool normalizeObservation_ = true;
  EigenVec obMean_;
  EigenVec obVar_;
  float obCount_ = 1e-4;
  EigenVec recentMean_, recentVar_, delta_;
  EigenVec epsilon;

  protected:
    std::vector<ChildEnvironment *> environments_;

};


class NormalDistribution {
 public:
  NormalDistribution() : normDist_(0.f, 1.f) {}

  float sample() { return normDist_(gen_); }
  void seed(int i) { gen_.seed(i); }

 private:
  std::normal_distribution<float> normDist_;
  static thread_local std::mt19937 gen_;
};
thread_local std::mt19937 raisim::NormalDistribution::gen_;


class NormalSampler {
 public:
  NormalSampler(int dim) {
    dim_ = dim;
    normal_.resize(THREAD_COUNT);
    seed(0);
  }

  void seed(int seed) {
    // this ensures that every thread gets a different seed
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < THREAD_COUNT; i++)
      normal_[0].seed(i + seed);
  }

  inline void sample(Eigen::Ref<EigenRowMajorMat> &mean,
                     Eigen::Ref<EigenVec> &std,
                     Eigen::Ref<EigenRowMajorMat> &samples,
                     Eigen::Ref<EigenVec> &log_prob) {
    int agentNumber = log_prob.rows();

#pragma omp parallel for schedule(auto)
    for (int agentId = 0; agentId < agentNumber; agentId++) {
      log_prob(agentId) = 0;
      for (int i = 0; i < dim_; i++) {
        const float noise = normal_[omp_get_thread_num()].sample();
        samples(agentId, i) = mean(agentId, i) + noise * std(i);
        log_prob(agentId) -= noise * noise * 0.5 + std::log(std(i));
      }
      log_prob(agentId) -= float(dim_) * 0.9189385332f;
    }
  }
  int dim_;
  std::vector<NormalDistribution> normal_;
};
 

template <class ChildEnvironment>
struct GenCoordFetcher: public raisim::VectorizedEnvironment<ChildEnvironment>{

  /*GenCoordFetcher(ChildEnvironment* env)
  : raisim::VectorizedEnvironment<ChildEnvironment>::environments_(env){ //ChildEnvironment* env = environments_[0]
    //cannot do this operation because world is a unique pointer. 
  }*/

    void getMotorTorques(Eigen::Ref<EigenVec> &tau){
       raisim::VectorizedEnvironment<ChildEnvironment>::environments_[0]->getMotorTorques(tau);
    }

    void getpTarget(Eigen::Ref<EigenVec> &pTarget){
         raisim::VectorizedEnvironment<ChildEnvironment>::environments_[0]->getReference(pTarget);
    }

    void getJointPositions(Eigen::Ref<EigenVec> &q){
         raisim::VectorizedEnvironment<ChildEnvironment>::environments_[0]->getJointPositions(q);
    }

    void getJointVelocities(Eigen::Ref<EigenVec> & dotq){
         raisim::VectorizedEnvironment<ChildEnvironment>::environments_[0]->getJointVelocities(dotq);
    }

    void getActualTorques(Eigen::Ref<EigenVec> &tau){
         raisim::VectorizedEnvironment<ChildEnvironment>::environments_[0]->getActualTorques(tau);
    }

    void getJointAccelerations(Eigen::Ref<EigenVec> &acc){
         raisim::VectorizedEnvironment<ChildEnvironment>::environments_[0]->getJointAccelerations(acc);
    }


    //Just got the access to the protected member. I couldn't copy that element into another because of unique_ptr

    ChildEnvironment* env_;

};



struct VariablesPlot: private raisim::ENVIRONMENT{


  void getActualTorques(Eigen::Ref<EigenVec>& tau){
		tau = ENVIRONMENT::computedTorques.cast<float>();
	}

	void getMotorTorques(Eigen::Ref<EigenVec>& tau){
		tau = ENVIRONMENT::motorTorque.cast<float>();
	}

	void getReference(Eigen::Ref<EigenVec>& tau){
		tau = ENVIRONMENT::pTarget_.tail(3).cast<float>();
	}

	void getJointPositions(Eigen::Ref<EigenVec>& q){
		q = ENVIRONMENT::gc_.tail(3).cast<float>();
	}
	
	void getJointVelocities(Eigen::Ref<EigenVec>& dotq){
		dotq = ENVIRONMENT::gv_.tail(3).cast<float>();
	}

	void getJointAccelerations(Eigen::Ref<EigenVec>& ddotq){
		ddotq = ga_.cast<float>();
	}

};




}

#endif //SRC_RAISIMGYMVECENV_HPP
