//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include "Eigen/Eigen"

#include <cstdlib>
#include "raisim/World.hpp"
#include <fstream>
#include <vector> 

#include "../../RaisimGymEnv.hpp"
#include "raisim/contact/Contact.hpp"

#include "Actuators.hpp"

#include <fcntl.h>
#include <termios.h>
 
#include <stdio.h>
#include "RandomNumberGenerator.hpp"

#define ESC_ASCII_VALUE                 0x1b
namespace raisim {
	
//#define num_row_in_file 4000
//#define decimal_precision 8
class ENVIRONMENT : public RaisimGymEnv {

 public:

	explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
		RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

		/// create world
		world_ = std::make_unique<raisim::World>();
		hebiMotor_ = std::make_shared<Actuators>();
		home_path_ = "/home/claudio/raisim_ws/raisimlib";
		/// add objects
		prisma_walker = world_->addArticulatedSystem(home_path_ + "/rsc/prisma_walker/urdf/prisma_walker.urdf");
		prisma_walker->setName("prisma_walker");
		prisma_walker->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
		world_->addGround();
		Eigen::Vector3d gravity_d(0.0,0.0,0);
		/// get robot data
		gcDim_ = prisma_walker->getGeneralizedCoordinateDim();
		gvDim_ = prisma_walker->getDOF();
		nJoints_ = gvDim_ - 6;

		/// initialize containers
		gc_.setZero(gcDim_); gc_init_.setZero(gcDim_); gc_noise_.setZero(gcDim_);
		gv_.setZero(gvDim_); gv_init_.setZero(gvDim_); gv_noise_.setZero(gvDim_);
		pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget3_.setZero(nJoints_);  
		linkTorque_.setZero(gvDim_);

		roll_init_= 0.0;pitch_init_=0.0;yaw_init_=0.0;
		q_ = Eigen::AngleAxisf(roll_init_, Eigen::Vector3f::UnitX())
		* Eigen::AngleAxisf(pitch_init_, Eigen::Vector3f::UnitY())
		* Eigen::AngleAxisf(yaw_init_, Eigen::Vector3f::UnitZ());
		/// this is nominal configuration of prisma walker
		gc_init_ << 0.0, 0.0, 0.33, q_.w(), q_.x(), q_.y(), q_.z(), 0.6, 0.6, 0.0; //(pos, orientament, configurazione)

		/// set pd gains
		Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
		jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(8);
		jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.05);
		//usando un d gain 0.2 non converge a niente, mettendolo a 2 invece per un po' sembra migliorare ma poi torna a peggiorare
		int max_time;
		READ_YAML(int, num_seq, cfg_["num_seq"]);
   		READ_YAML(int, num_seq_vel, cfg_["num_seq_vel"]);
		READ_YAML(int, max_time, cfg_["max_time"]);  //mu,Step = control_dt /max time
		READ_YAML(bool, sea_included_, cfg_["sea_included"]);
		READ_YAML(bool, use_privileged_, cfg_["use_privileged"]);

		num_step = max_time/control_dt_;
		historyPosLength_ = nJoints_*num_seq;
		historyVelLength_ = nJoints_*num_seq_vel;
		joint_history_pos_.setZero(historyPosLength_);
    		joint_history_vel_.setZero(historyVelLength_);

		current_action_.setZero(3);
		index_imitation_ = 0;
		if(!sea_included_)
			prisma_walker->setPdGains(jointPgain, jointDgain);
		else
			prisma_walker->setPdGains(Eigen::VectorXd::Zero(gvDim_), Eigen::VectorXd::Zero(gvDim_));
		prisma_walker->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

		/// MUST BE DONE FOR ALL ENVIRONMENTS
		obDim_ = historyPosLength_ + historyVelLength_ + current_action_.size();
		int privileged_obs_dim = 9 + 6 + 2 + 1 + 1;
		//pitch + yaw + bodyLinearVel + bodyAngularVel + currentImitationError + bodyHeight + footState
		//+nextMotorPos	
		actionDim_ = nJoints_; 
		
		actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
		if(use_privileged_){
			obDim_ += privileged_obs_dim;
		}
		obDouble_.setZero(obDim_);

		/// action scaling
		actionMean_ = gc_init_.tail(nJoints_);
		double action_std;

		command_<< 0.2, 0, 0.12;
		READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
		
		// Open port
		actionStd_.setConstant(action_std);
		/// Reward coefficients
		rewards_.initializeFromConfigurationFile (cfg["reward"]);

		footIndex_ = prisma_walker->getBodyIdx("link_3");
		
		foot_center_ = prisma_walker->getFrameIdxByLinkName("piede_interno"); //since there is a fixed joint, I cant assign a body idx to it
		foot_sx_ = prisma_walker->getFrameIdxByLinkName("piede_sx");
		foot_dx_ = prisma_walker->getFrameIdxByLinkName("piede_dx");

		m1_pos_.setZero(traj_size);
		m2_pos_.setZero(traj_size); 
		/// visualize if it is the first environment
		if (visualizable_) {
			server_ = std::make_unique<raisim::RaisimServer>(world_.get());
			server_->launchServer();
			server_->focusOn(prisma_walker);
 		}

		num_episode_ = 0;
		curr_imitation_ = 1e-6;
		actual_step_ = 0;
	 	clearance_foot_ = 0.0; //penalita' iniziale. Altrimenti non alzera' mai il piede
		//motors = new Actuators();
		//motors->initHandlersAndGroup(ActuatorConnected_, num_seq, num_seq_vel, visualizable_);
		openFile();

		for(int i = 0; i < prisma_walker->getMass().size(); i++){
      		realMass_.push_back(prisma_walker->getMass()[i]);	//it has 4 links (it takes into account only the bodies with not fixed joints)
			//std::cout<<realMass_[i]<<std::endl;
			//the link 0 is the base with the fixed legs m = 0.951376
			//link 1 is that attached to the first hebi m = 0.24
			//link 2 is that attached to the second hebi m = 0.1175
			//link 3 is the foot attached to the dynamixel m = 0.286
			//valgrind --leak-check=yes raisimGymTorch/env/bin/./debug /home/dev/raisim_ws/raisimlib/rsc/ /home/dev/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/cfg.yaml 

		}
		
		prisma_walker->getCollisionBody("piede_interno/0").setMaterial("foot");

		currRand_jointPos_.setZero(gcDim_); // No need to randomize the measure of position (so far)
		currRand_jointVel_.setZero(gvDim_); // 6 elements
		disturbanceFactor_ = 0.005;

		theta = Eigen::Vector3d::Zero(); dotTheta = Eigen::Vector3d::Zero(); motorTorque = Eigen::Vector3d::Zero();
		B_inverse = Eigen::Matrix3d::Zero();
		float gearRatio_square = 80.2*80.2;

		float motorInertia = 0.0043;
		B_inverse.block(0,0,2,2) << 1/(motorInertia*gearRatio_square), 0.0,
									0.0, 1/(motorInertia*gearRatio_square);

		nonLinearTerms = Eigen::Vector3d::Zero();
		nonLinearTerms_vecDyn_.setZero(gvDim_);
	}

	void openFile(){

		std::fstream m1_traj;
    		std::fstream m2_traj;

		m1_traj.open(home_path_ + "/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/pos_m1_18s.txt", std::ios::in);
    		m2_traj.open(home_path_ + "/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/pos_m2_18s.txt", std::ios::in);
	
		Eigen::VectorXd m1_pos(traj_size);
		Eigen::VectorXd m2_pos(traj_size);

		if(m1_traj.is_open() && m2_traj.is_open()){
			for(int j = 0;j<traj_size;j++){
				m1_traj >> m1_pos(j);  //one character at time store the content of the file inside the vector

				m1_pos_=-m1_pos;

				m2_traj >> m2_pos(j);

				m2_pos_=-m2_pos;
			}
		}
		else
			std::cout<<"File not opend!"<<std::endl;
		
		m1_traj.close(); 
		m2_traj.close();
	}


	void command_vel(double v_x, double v_y, double omega_z){ //This function can be called to declare a new command velocity
		command_[0] = v_x;
		command_[1] = v_y;
		command_[2] = omega_z;
		std::cout<<"command_vel: "<<command_<<std::endl;
	}

	void init() final { 
		//Enable torque: activate 
		//int dxl_comm_result = packetHandler_->write1ByteTxRx(portHandler_, dxl_id, addrTorqueEnable, TorqueEnable, &dxl_error_);

	}    

	void getActualTorques(Eigen::Ref<EigenVec>& tau){
		tau = computedTorques.cast<float>();
	}

	void getMotorTorques(Eigen::Ref<EigenVec>& tau){
		tau = motorTorque.cast<float>();
	}

	void getReference(Eigen::Ref<EigenVec>& tau){
		tau = pTarget_.tail(3).cast<float>();
	}

	void getJointPositions(Eigen::Ref<EigenVec>& q){
		q = gc_.tail(3).cast<float>();
	}
	
	void getJointVelocities(Eigen::Ref<EigenVec>& dotq){
		dotq = gv_.tail(3).cast<float>();
	}
	

	void reset() final {
		motorSpring_ = 20 + rn_.sampleUniform01()*180; //min and max of the spring coefficient

		setFriction();
		previous_height_ = 0;
		alfa_z_ = 0;
		max_height_ = 0;
		
		if((fallen_ && rn_.intRand(0,3) == 2) || rn_.intRand(0,9) == 1){
			index_imitation_ = traj_size*rn_.sampleUniform01();

			gc_init_[7] = m1_pos_(index_imitation_ - 1);
			gc_init_[8] = m2_pos_(index_imitation_ - 1);
		}

		nonLinearTerms_vecDyn_ = prisma_walker->getNonlinearities(world_->getGravity());
		nonLinearTerms = nonLinearTerms_vecDyn_.e().tail(3);
		theta(0) = gc_init_[7] + (1/motorSpring_)*nonLinearTerms(0); 
		theta(1) = gc_init_[8] + (1/motorSpring_)*nonLinearTerms(1);
	
		theta(2) = gc_init_(9);	

		dotTheta = Eigen::Vector3d::Zero();

		prisma_walker->setState(gc_init_, gv_init_);
		updateObservation();
		computedTorques.setZero();
	}
	
 
	float step(const Eigen::Ref<EigenVec>& action) final {
		/// action scaling
		pTarget3_ = action.cast<double>(); //dim=n_joints
		pTarget3_ = pTarget3_.cwiseProduct(actionStd_);
		actionMean_ << m1_pos_(index_imitation_), m2_pos_(index_imitation_), 0.0;
		pTarget3_ += actionMean_;
		pTarget_.tail(nJoints_) << pTarget3_;
		current_action_ = pTarget3_;
	
		if(!sea_included_){
			//prisma_walker->setPdTarget(pTarget_, vTarget_);

			prisma_walker->setPdGains(Eigen::VectorXd::Zero(gvDim_), Eigen::VectorXd::Zero(gvDim_));
			motorTorque = 80*(pTarget3_ - gc_.tail(3)) - 0.8*gv_.tail(3);
			for(int i = 0; i< motorTorque.size(); i++){
				motorTorque(i) = std::clamp(motorTorque[i], -gearRatio*7, gearRatio*7);
			}
			linkTorque_.tail(3) = motorTorque;
			computedTorques = prisma_walker->getGeneralizedForce().e().tail(3);

		}
		else{
			motorTorque = 80*(pTarget3_ - theta) - 0.9*dotTheta;

			/*for(int i = 0; i< motorTorque.size(); i++){
				motorTorque(i) = std::clamp(motorTorque[i], -gearRatio*7, gearRatio*7);
			}*/
			
			dotTheta += control_dt_*B_inverse*(motorTorque - motorSpring_*(theta - gc_.tail(3)));
			theta += control_dt_*dotTheta;

			if(dotTheta.squaredNorm() == 0){
				nonLinearTerms_vecDyn_ = prisma_walker->getNonlinearities(world_->getGravity());
				nonLinearTerms = nonLinearTerms_vecDyn_.e().tail(3);
				theta = gc_.tail(3) + (1/motorSpring_)*nonLinearTerms;
			}

			theta(2) = gc_(9);
			dotTheta(2) = gv_(8);

			linkTorque_.tail(3) << motorSpring_*(theta - gc_.tail(3));
		}
		
		
		pTarget_.tail(nJoints_) << std::sin(2*M_PI*0.1*t), 0, 0;
		t+=control_dt_;
		
		Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
		jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(80);
		jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.5);
		prisma_walker->setPdGains(jointPgain, jointDgain);
		prisma_walker->setPTarget(pTarget_);
		

		//prisma_walker->setGeneralizedForce(pTarget_);
		if(ActuatorConnected_){
			motors->sendCommand(linkTorque_, sea_included_);
		}

		for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
			if(server_) server_->lockVisualizationServerMutex();
			world_->integrate();
			if(server_) server_->unlockVisualizationServerMutex();
		}

		updateObservation();
	
		contacts(); //va dopo il world integrate, perche' il world integrate aggiorna i contatti. 
		incrementIndices();
		imitation_function();
		external_force();

		rewards_.record("torque", prisma_walker->getGeneralizedForce().squaredNorm());
		rewards_.record("Joint_velocity", joint_history_vel_.tail(3).squaredNorm());
		//rewards_.record("Joint_accelerations",
				 //((joint_history_vel_.tail(3) - joint_history_vel_.segment(0,3))/control_dt_).squaredNorm());

		rewards_.record("imitation", std::exp(-2*(1/sigma_square)*(error_m1_*error_m1_ + error_m2_*error_m2_)));
		//rewards_.record("dynamixel_joint", std::exp(-2*(1/sigma)*gc_[9]*gc_[9]));
		rewards_.record("angular_penalty", bodyAngularVel_[0] + bodyAngularVel_[1]); //+0.025*bodyLinearVel_[2]
		rewards_.record("slip", curr_fact_slip_*slip_term_);
		rewards_.record("ground_clearence", clearance_foot_);

		if(cF_["center_foot"] == 0)
			rewards_.record("BodyMovementWithLateralFeet", bodyLinearVel_.squaredNorm() + bodyAngularVel_.squaredNorm());

		actual_step_++;	
		if(actual_step_ == num_step){
			gc_init_[7] = gc_[7];
			gc_init_[8] = gc_[8];
		}

		//std::cout<<slip_term_<<std::endl;

		return rewards_.sum();
		
	}

	void incrementIndices(){

		index_imitation_++;

		if(index_imitation_ >= traj_size){
			index_imitation_ = 0;
		}
	}
	
	void discardIndices(){
		double distance_m1 = 0;
		double distance_m2 = 0;
		do{
			index_imitation_ ++;
			distance_m1 = std::abs(m1_pos_(index_imitation_) - m1_pos_(index_imitation_ + 1));
			distance_m2 = std::abs(m2_pos_(index_imitation_) - m2_pos_(index_imitation_ + 1));
		}while( distance_m1 < curr_imitation_ && distance_m2 < curr_imitation_);
	}

	void imitation_function(){
		if(sea_included_){
			error_m1_ = theta(0) - m1_pos_(index_imitation_);
			error_m2_ = theta(1) - m2_pos_(index_imitation_);
		}else{
			error_m1_ = gc_[7] - m1_pos_(index_imitation_);
			error_m2_ = gc_[8] - m2_pos_(index_imitation_);
		}
	}
 

	void clearance(){
	
		if(bodyFootPos_[0] >= posForFootHitGround_){
			curr_fact = 1;
			if(previous_height_ > footPosition_[2]){ 
				clearance_foot_ = std::exp(-50*(previous_height_ - targetHeight_)*(previous_height_ - targetHeight_));
			}
			else{
				previous_height_ = footPosition_[2];
				curr_fact = 2;
			}
		} 

		if(cF_["center_foot"]==1){
			previous_height_ = 0;
		}

	}


	void slippage(){
		
		slip_term_ = 0;
		if(cF_["center_foot"] == 1 || footPosition_[2] < 0.001){
			slip_term_ = std::sqrt(vel_[0]*vel_[0] + vel_[1]*vel_[1]);
		}	

		if(cF_["center_foot"] == 1 && cF_["lateral_feet"] == 1 && index_imitation_ > 800 && index_imitation_ < 1600){
			if(slip_term_ > 0)
				posForFootHitGround_ -= 0.01;
				curr_fact_slip_ += 0.01;
		}
	}

    void swapMatrixRows(Eigen::Matrix3d &mat){		
		Eigen::Vector3d temp;
		//riga x-> z
		temp = mat.row(2); 
		mat.row(2) = mat.row(0);
		//riga y>x
		mat.row(0) = mat.row(1);
		//row z->y
		mat.row(1) = temp;
    }


	void contacts(){
		
		prisma_walker->getFrameVelocity(foot_center_, vel_);

		cF_["center_foot"] = 0;
		cF_["lateral_feet"] = 0;
		for(auto& contact: prisma_walker->getContacts()){
			if (contact.skip()) continue;  //contact.skip() ritorna true se siamo in una self-collision, in quel caso ti ritorna il contatto 2 volte
			if(contact.getlocalBodyIndex() == 3 ){
				cF_["center_foot"] = 1;
			} 
			if(contact.getlocalBodyIndex() == 0 ){  
				cF_["lateral_feet"] = 1; 
			}      
		}

		slippage();
		clearance();
	} 

	inline void computeSwingTime(){

		if(cF_["center_foot"] == 1){  //NON TENERE LE COSE DENTRO IL FOR, PERCHè Altrimenti chiama le stesse funzioni piu' VOLTE!!!
			swing_time_ = std::chrono::duration<double, std::milli>(0);	
			lift_instant_ = std::chrono::steady_clock::now();
		}
	
		if(cF_["center_foot"] == 0){
			land_instant_ = std::chrono::steady_clock::now();
			swing_time_ += std::chrono::duration<double, std::milli>(land_instant_ - lift_instant_);
			lift_instant_ = std::chrono::steady_clock::now();
		}
	}


	void updateObservation() {
		
		std::vector<double> nextMotorPos;
		int indexToIncrease = index_imitation_;
		for(int i = 0; i<5; i++){
			indexToIncrease ++;
			if(indexToIncrease> traj_size)
				indexToIncrease = indexToIncrease - traj_size;
			nextMotorPos.push_back(m1_pos_(indexToIncrease));
			nextMotorPos.push_back(m2_pos_(indexToIncrease));
		}
		
		Eigen::Map<Eigen::VectorXd> nextMotorValues(nextMotorPos.data(), nextMotorPos.size());

		//Il giunto di base e' fisso rispetto alla base. Quindi l'orientamento della base e' quello del motore 
		if(ActuatorConnected_){ 	//If motors are not connected you get a seg fault om the sendRequest
			Eigen::VectorXd obs_motors = motors->getFeedback();
 			obDouble_ << obs_motors,
			current_action_; 
		}else{
			prisma_walker->getState(gc_, gv_);//generalized coordinate generalized velocity wrt initial coordinate gc_init
			
			{//clean measures to construct bodyFootPos
				quat_[0] = gc_[3]; 
				quat_[1] = gc_[4]; 
				quat_[2] = gc_[5]; 
				quat_[3] = gc_[6];

				raisim::quatToRotMat(quat_, rot_); 
				bodyPos_ = rot_.e().transpose() * gc_.segment(0, 3);  //position of the robot reported to the body frame
				prisma_walker->getFramePosition(foot_center_, footPosition_);
				bodyFootPos_ = rot_.e().transpose() * footPosition_.e(); //position of the foot reported to the body frame
				bodyFootPos_ = bodyFootPos_ - bodyPos_; //position of the foot with respect the body
				//if I consider only this term "bodyFootPos_ = rot_.e().transpose() * footPosition_.e();", it changes only when I move the foot, it must be changed also when the robot moves the body
			}

			updateJointHistory();

			quat_[0] = gc_noise_[3]; 
			quat_[1] = gc_noise_[4]; 
			quat_[2] = gc_noise_[5]; 
			quat_[3] = gc_noise_[6];

			raisim::quatToRotMat(quat_, rot_randomized_); 
		
			bodyLinearVel_ = rot_randomized_.e().transpose() * gv_noise_.segment(0, 3); //linear velocity reported to the base frame
			bodyAngularVel_ = rot_randomized_.e().transpose() * gv_noise_.segment(3, 3);

			error_m1_obs_ = gc_noise_[7] - m1_pos_(index_imitation_);
			error_m2_obs_ = gc_noise_[8] - m2_pos_(index_imitation_);

			if(use_privileged_)
				obDouble_ <<rot_randomized_.e().row(0).transpose(),
				rot_randomized_.e().row(1).transpose(),
				rot_randomized_.e().row(2).transpose(), /// body orientation e' chiaro che per la camminata, e' rilevante sapere come sono orientato rispetto all'azze z, e non a tutti e 3. L'orientamento rispetto a x e' quanto sono chinato in avanti, ma io quando cammino scelgo dove andare rispetto a quanto sono chinato??? ASSOLUTAMENTE NO! Anzi, cerco di non essere chinato. Figurati un orentamento rispetto a y, significherebbe fare la ruota sul posto /// body linear&angular velocity
				bodyAngularVel_, bodyLinearVel_,
				error_m1_obs_,
				error_m2_obs_,
				gc_[2],
				cF_["center_foot"],
				joint_history_pos_, /// joint angles
				joint_history_vel_,
				current_action_;
			else
				obDouble_<<joint_history_pos_,
					joint_history_vel_,
					current_action_;
		}
	}


	void updateJointHistory(){
		
		Eigen::VectorXd actualJointPos, actualJointVel;
		actualJointPos.setZero(gcDim_); actualJointVel.setZero(gvDim_);

		if(sea_included_){
			actualJointPos << gc_.segment(0,7), theta;
			actualJointVel << gv_.segment(0,6), dotTheta;
		}else{
			actualJointPos << gc_;
			actualJointVel << gv_;
		}

		joint_history_pos_.head(historyPosLength_ - 3) = joint_history_pos_.tail(historyPosLength_ - 3);
		joint_history_vel_.head(historyVelLength_ - 3) = joint_history_vel_.tail(historyVelLength_ - 3);
		
		for(int i = 0; i < currRand_jointPos_.size(); i++)
			gc_noise_[i] = actualJointPos[i] + currRand_jointPos_[i]*rn_.sampleUniform(); 

		for(int i = 0; i < currRand_jointVel_.size(); i++)
			gv_noise_[i] = actualJointVel[i] + currRand_jointVel_[i]*rn_.sampleUniform();	

		joint_history_pos_.tail(3) = gc_noise_.tail(3);
		joint_history_vel_.tail(3) = gv_noise_.tail(3);

		//Reshaping della time series
		/*for(int i = 0; i< 3; i++){
			for(int j = 0; j<num_seq; j++){
				joint_history_pos_reshaped_(Eigen::seq(i*num_seq, (i+1)*(num_seq-1)))[j] = joint_history_pos_(Eigen::seq(j*3, (j+1)*3))[i];
			}
		}

		for(int i = 0; i< 3; i++){
			for(int j = 0; j<num_seq_vel; j++){
				joint_history_vel_reshaped_(Eigen::seq(i*num_seq_vel, (i+1)*(num_seq_vel-1)))[j] = joint_history_vel_(Eigen::seq(j*3, (j+1)*3))[i];
			}
		}*/
	}


	void observe(Eigen::Ref<EigenVec> ob) final {
		/// convert it to float
		ob = obDouble_.cast<float>();
		//std::cout << "ob double: " << ob.size() << std::endl;
	}


	bool isTerminalState(float& terminalReward) final {
		terminalReward = float(terminalRewardCoeff_);
		
		Eigen::Vector3d z_vec = Eigen::Vector3d::Zero();
		Eigen::Vector3d z_axis(0,0,1);
		z_vec = rot_.e().row(2).transpose();
		alfa_z_ = acos(z_vec.dot(z_axis));
		alfa_z_ = (alfa_z_*180)/M_PI;
		 
		if (std::sqrt(error_m1_*error_m1_ + error_m2_*error_m2_) > 3*sigma || alfa_z_>18){
			fallCount_++;
			index_imitation_ -= 100;
			if(index_imitation_ < 0){
				index_imitation_ = std::abs(index_imitation_ - traj_size);
			}
			gc_init_[7] = m1_pos_(index_imitation_);
			gc_init_[8] = m2_pos_(index_imitation_);
			error_penalty_ += 0.4;
			fallen_ = true;
			//return true;
		}		
	
		terminalReward = 0.f;
		return false;
	}

	void curriculumUpdate() {
		//generate_command_velocity(); //La metto qui perche' la reset viene chiamata troppe volte

		if(num_episode_ > 10 && !fallen_){
			curr_imitation_ += 1e-6; 
		}

		if(fallen_ && curr_imitation_ > 1e-6){
			curr_imitation_ -= 1.5e-6;
		}

		if(curr_imitation_ > 1e-2)
			curr_imitation_ = 1e-2;
		if(curr_imitation_ < 1e-6)
			curr_imitation_ = 1e-6;

		num_episode_++;

		if(visualizable_){
			std::cout<<"Curr imitation   "<<curr_imitation_<<std::endl;
		}
		//IT must go at the end of the episode
		actual_step_ = 0;
		error_penalty_ = 0;


		if(num_episode_ > 1000){
			if(!fallen_){
				if(disturbanceGone_ = true){
					disturbanceFactor_ += 0.0025;
					disturbanceGone_ = false;
				}

				for(int i = 0; i < gcDim_; i++){
					if(currRand_jointPos_[i] < 0.5)
						currRand_jointPos_[i] += 0.001;
				}

				for(int i = 0; i < gvDim_; i++){
					if(currRand_jointVel_[i] < 1)
						currRand_jointVel_[i] += 0.001;
				}
			}
			else{
				reduceRand();
			}
		}

		if(fallen_){
			RSINFO_IF(visualizable_, "fall count = " << fallCount_)
		}
		fallen_ = false;
		fallCount_ = 0;
	};

	void setFriction(){
		double friction_coefficient = 0;
		if(num_episode_ < 200){   //if else are better for boolean. WHen you have nested conditions like this use switch
			friction_coefficient = 0.9;
			prisma_walker->getMass()[0] = realMass_[0] + 0.07*rn_.sampleUniform() + 0.2; //add 100g of te base motor
			prisma_walker->getMass()[1] = realMass_[1] + 0.03*rn_.sampleUniform() + 0.2; //add 100g of te base motor
			prisma_walker->getMass()[2] = realMass_[2] + 0.02*rn_.sampleUniform() + 0.08; //add 100g of te base motor 
			prisma_walker->getMass()[3] = realMass_[3] + 0.04*rn_.sampleUniform();
		}
		else if (num_episode_ < 500){
			friction_coefficient = 0.75 + rn_.sampleUniform01() * 0.15; //linear interpolation
			prisma_walker->getMass()[0] = realMass_[0] + 0.18*rn_.sampleUniform() + 0.2; //add 100g of te base motor
			prisma_walker->getMass()[1] = realMass_[1] + 0.08*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.03*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.09*rn_.sampleUniform();
		}
		else if (num_episode_ < 800){
			friction_coefficient = 0.5 + rn_.sampleUniform01() * 0.4;
			prisma_walker->getMass()[0] = realMass_[0] + 0.30*rn_.sampleUniform() + 0.2; //add 100g of te base motor
			prisma_walker->getMass()[1] = realMass_[1] + 0.12*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.04*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.15*rn_.sampleUniform();
		}
		else if (num_episode_ < 1100){
			friction_coefficient = 0.3 + rn_.sampleUniform01() * 0.2;
			prisma_walker->getMass()[0] = realMass_[0] + 0.40*rn_.sampleUniform() + 0.2; //add 100g of te base motor
			prisma_walker->getMass()[1] = realMass_[1] + 0.18*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.05*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.20*rn_.sampleUniform();
		}
		else if (num_episode_ <= 1600){
			friction_coefficient = 0.25 + rn_.sampleUniform01() * 0.75; //[0,25,1]
			prisma_walker->getMass()[0] = realMass_[0] + 0.45*rn_.sampleUniform() + 0.2; //add 100g of te base motor
			prisma_walker->getMass()[1] = realMass_[1] + 0.20*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.07*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.23*rn_.sampleUniform();
		}
		else if (num_episode_ > 1600){
			friction_coefficient = 0.25 + rn_.sampleUniform01() * 0.75; //[0,25,1]
			prisma_walker->getMass()[0] = realMass_[0] + 0.55*rn_.sampleUniform() + 0.2; //add 100g of te base motor
			prisma_walker->getMass()[1] = realMass_[1] + 0.22*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.09*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.25*rn_.sampleUniform();
		}
		/*0.951376
		0.241507
		0.1175
		0.284909*/

		prisma_walker -> updateMassInfo();
		world_->setMaterialPairProp("ground", "foot", friction_coefficient, 0.0, 0.001); //mu, bouayancy, restitution velocity (and minimum impact velocity to make the object bounce) 
		
	}

	void reduceRand(){

		if(disturbanceFactor_ > 0.01)
			disturbanceFactor_ -= 0.005;
		
		for(int i = 0; i < gcDim_; i++){
			if(currRand_jointPos_[i] > 0) currRand_jointPos_[i] -= 0.001;
		}

		for(int i = 0; i < currRand_jointVel_.size(); i++){
			if(currRand_jointVel_[i] > 0) currRand_jointVel_[i] -= 0.001;	
		}

		int indexPos = rn_.intRand(0,9);
		int indexVel = rn_.intRand(0,8);

		if(currRand_jointPos_[indexPos] < 2)
			currRand_jointPos_[indexPos] += 0.001;

		if(currRand_jointPos_[indexVel] < 2)
			currRand_jointVel_[indexVel] += 0.001;
	}

	void external_force(){
 		raisim::Vec<3> disturbance;
		if(rn_.intRand(0,10) == 5){
			int bodyIndexSubjet = rn_.intRand(0,3);
			disturbanceGone_ = true;
			float ang = M_PI * rn_.sampleUniform01();
			
			disturbance[0] = disturbanceFactor_ * std::cos(ang);  //tp distribute the force
			disturbance[1] = disturbanceFactor_ * std::sin(ang);
			disturbance[2] = 0.001*rn_.sampleUniform();
			prisma_walker->setExternalForce(bodyIndexSubjet, disturbance);
		}
		
	}


	private:
		std::string home_path_;
		int gcDim_, gvDim_, nJoints_,timing_,fbk_counter_,n_campione_vel_,n_campione_pos_, index_imitation_;
		float alfa_z_, roll_init_, pitch_init_, yaw_init_; // initial orientation of prisma prisma walker
		Eigen::VectorXd m1_pos_;
		Eigen::VectorXd m2_pos_;
 
		double smoothing_factor_ = 0.06;
		raisim::Mat<3,3> rot_randomized_;
		raisim::Vec<4> quaternion_;
 
		Eigen::VectorXd filtered_acc_ = Eigen::VectorXd::Zero(3);
 
 		Eigen::Quaternionf q_;
		double mean_value_x_ = 0.0;double mean_value_y_ = 0.0;double mean_value_z_ = 0.0;
		Eigen::VectorXd real_lin_acc_ = Eigen::VectorXd::Zero(3);

		Eigen::Matrix3d R_imu_base_frame_ = Eigen::Matrix3d::Zero(3,3);
		Eigen::Matrix3d real_orientation_matrix_ = Eigen::Matrix3d::Zero(3,3);
		Eigen::Vector3d rpy_;
		bool visualizable_ = false;
		raisim::ArticulatedSystem* prisma_walker;
		Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, gc_noise_, gv_noise_, pTarget_, pTarget3_, vTarget_;
		const int terminalRewardCoeff_ = -12.;
		Eigen::VectorXd actionMean_, actionStd_, obDouble_;
		Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, bodyAngularVel_real_, bodyFootPos_, bodyPos_, initBodyPos_, initFootPos_;
		std::ofstream v_lin_sim_ = std::ofstream("v_lin_sim_40.txt");
		std::ofstream p_sim_ = std::ofstream("p_sim_40.txt");
		std::ofstream ori_sim_ = std::ofstream("ori_sim_40.txt");
		std::ofstream v_ang_sim_ = std::ofstream("v_ang_sim_40.txt");
		raisim::Vec<4> quat_;
		raisim::Mat<3,3> rot_;
		size_t foot_center_, footIndex_, foot_sx_ ,foot_dx_, contact_;
		raisim::CoordinateFrame footframe_,frame_dx_foot_,frame_sx_foot_;
		raisim::Vec<3> vel_, footPosition_;
		/// these variables are not in use. They are placed to show you how to create a random number sampler.
		std::normal_distribution<double> normDist_;
		RandomNumberGenerator<float> rn_;
		Actuators *motors;
		std::shared_ptr<Actuators> hebiMotor_;

		double slip_term_;
		double previous_height_, max_height_, clearance_foot_ = 0;
		bool max_clearence_ = false;
		Eigen::Vector3d command_;

		std::chrono::duration<double, std::milli> elapsed_time_; 
		std::chrono::duration<double, std::milli> swing_time_; 
		std::chrono::steady_clock::time_point begin_, end_, lift_instant_, land_instant_;
		std::map<std::string,int> cF_ = {
			{"center_foot", 0},
			{"lateral_feet", 0},
		};
		int num_seq, num_seq_vel, num_step;
		Eigen::VectorXd joint_history_pos_, joint_history_vel_, current_action_;
		Eigen::VectorXd joint_history_pos_reshaped_, joint_history_vel_reshaped_;
		double H_ = 0.0;
		int num_episode_ = 0;
		
		double curr_imitation_, motorSpring_; 
		bool fallen_ = false;
		int actual_step_;
		double error_m1_ = 0, error_m2_ = 0, error_m1_obs_ = 0, error_m2_obs_ = 0;
		const float sigma = 0.4;
		float sigma_square = sigma*sigma;
		Eigen::VectorXd nextMotorPositions_;
		const int traj_size = 1818;
		bool ActuatorConnected_ = false;
		int curr_fact = 1;
		float targetHeight_ = 0.11;
		float error_penalty_ = 0;
		int posForFootHitGround_ = 0;
		float curr_fact_slip_ = 1;

		std::vector<double> realMass_;
		Eigen::VectorXd currRand_jointPos_;
		Eigen::VectorXd currRand_jointVel_;
		float disturbanceFactor_;
		std::ofstream torques;
		int fallCount_ = 0;
		bool disturbanceGone_ = false;

		Eigen::VectorXd linkTorque_;
		Eigen::Vector3d theta, dotTheta, motorTorque;
		Eigen::Matrix3d B_inverse;
		bool sea_included_;
		Eigen::Vector3d computedTorques;
		const double gearRatio = 762.222; //std::clamp wnats all the lement of the same type
		Eigen::Vector3d nonLinearTerms;
		VecDyn nonLinearTerms_vecDyn_;

		bool use_privileged_;

		int historyPosLength_;
		int historyVelLength_;

		float t = 0;


};
//thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

