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
#include <boost/algorithm/string.hpp>
#include <stdio.h>
#include "RandomNumberGenerator.hpp"

#define ESC_ASCII_VALUE                 0x1b
namespace raisim {
	
//#define num_row_in_file 4000
//#define decimal_precision 8

enum TerrainType {
  Flat,
  SingleStep
};


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
		world_->addGround(0.0, "ground");

		/// get robot data
		gcDim_ = prisma_walker->getGeneralizedCoordinateDim();
		gvDim_ = prisma_walker->getDOF();
		nJoints_ = gvDim_ - 6;

		/// initialize containers
		gc_.setZero(gcDim_); gc_init_.setZero(gcDim_); gc_noise_.setZero(gcDim_);
		gv_.setZero(gvDim_); gv_init_.setZero(gvDim_); gv_noise_.setZero(gvDim_);
		ga_.setZero(nJoints_);
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
		jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(20);
		jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.5);
		//usando un d gain 0.2 non converge a niente, mettendolo a 2 invece per un po' sembra migliorare ma poi torna a peggiorare
		int max_time;
		READ_YAML(int, num_seq, cfg_["num_seq"]);
   		READ_YAML(int, num_seq_vel, cfg_["num_seq_vel"]);
		READ_YAML(int, max_time, cfg_["max_time"]);  //mu,Step = control_dt /max time


		READ_YAML(bool, sea_included_, cfg_["sea_included"]);
		READ_YAML(bool, use_privileged_, cfg_["use_privileged"]);
		READ_YAML(bool, implicitIntegration, cfg_["implicitIntegration"]);
		
		numJointsControlled = 3;
		num_step = max_time/control_dt_;
		historyPosLength_ = numJointsControlled*num_seq;
		historyVelLength_ = numJointsControlled*num_seq_vel;
		joint_history_pos_.setZero(historyPosLength_);
    	joint_history_vel_.setZero(historyVelLength_);

		current_action_.setZero(3);
		index_imitation_ = 0;
		if(implicitIntegration)
			prisma_walker->setPdGains(jointPgain, jointDgain);
		else
			prisma_walker->setPdGains(Eigen::VectorXd::Zero(gvDim_), Eigen::VectorXd::Zero(gvDim_));
		
		prisma_walker->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

		/// MUST BE DONE FOR ALL ENVIRONMENTS
		obDim_ = 3 + 3 + 3 + 3;
		int privileged_obs_dim = 6 + 3;
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
		READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
		actionStd_.setConstant(action_std);
		actionStd_[2] = 0.4;

		/// Reward coefficients
		rewards_.initializeFromConfigurationFile (cfg["reward"]);

		foot_center_ = prisma_walker->getFrameIdxByLinkName("piede_interno"); //since there is a fixed joint, I cant assign a body idx to it

		m1_pos_.setZero(traj_size);
		m2_pos_.setZero(traj_size); 

		m1_vel_.setZero(traj_size);
		m2_vel_.setZero(traj_size);
		/// visualize if it is the first environment
		if (visualizable_) {
			server_ = std::make_unique<raisim::RaisimServer>(world_.get());
			server_->launchServer();
			server_->focusOn(prisma_walker);
 		}

		num_episode_ = 0;
		actual_step_ = 0;
	 	clearance_foot_ = 0.0; 
		footSlip_ = 0;
		angular_command_ = 0.0; 
		curr_imitation_ = 0.5;

		terrainType_ = TerrainType::Flat;
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

		prisma_walker->getFrameVelocity(foot_center_, Footvel_);
		angular_command_ = 0;
		footVelocityRef_ = 0;
	}

	void openFile(){

		std::fstream m1_traj;
    	std::fstream m2_traj;

		std::fstream m1_vel;
    	std::fstream m2_vel;


		m1_traj.open(home_path_ + "/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/pos_m1_18s.txt", std::ios::in);
    	m2_traj.open(home_path_ + "/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/pos_m2_18s.txt", std::ios::in);

		Eigen::VectorXd m1_pos(traj_size);
		Eigen::VectorXd m2_pos(traj_size);

		if(m1_traj.is_open() && m2_traj.is_open()){
			for(int j = 0;j<traj_size;j++){
				m1_traj >> m1_pos(j);  //one character at time store the content of the file inside the vector
				//m1_vel >> m1_vel_(j);

				m1_pos_=-m1_pos;
				m1_stdvector_.push_back(m1_pos(j));

				m2_traj >> m2_pos(j);
				//m2_vel >> m2_vel_(j);
				m2_pos_=-m2_pos;
				m2_stdvector_.push_back(m2_pos(j));

			}
		}
		else
			std::cout<<"File not opend!"<<std::endl;
		
		m1_traj.close(); 
		m2_traj.close();

	}


	void command_vel(double omega_z){ //This function can be called to declare a new command velocity

		angular_command_ = omega_z;
		std::cout<<"command_vel: "<<angular_command_<<std::endl;
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
		tau = pTarget3_.cast<float>();
	}

	void getJointPositions(Eigen::Ref<EigenVec>& q){
		q = gc_.tail(3).cast<float>();
		if(ActuatorConnected_){
			Eigen::VectorXd motorVariables;
			motorVariables.setZero(6);

			motorVariables = motors->dataPlotMotorVariables();

			q = motorVariables.segment(0,3).cast<float>();
		}

	}
	
	void getJointVelocities(Eigen::Ref<EigenVec>& dotq){
		dotq = gv_.tail(3).cast<float>();
		if(ActuatorConnected_){
			Eigen::VectorXd motorVariables;
			motorVariables.setZero(6);
			motorVariables = motors->dataPlotMotorVariables();
			dotq = motorVariables.segment(3,3).cast<float>();
		}
	}

	void getJointAccelerations(Eigen::Ref<EigenVec>& ddotq){
		ddotq = ga_.cast<float>();
	}

	void getPitch(Eigen::Ref<EigenVec>& pitch){
		pitch = rot_randomized_.e().row(1).transpose().cast<float>();
		if(ActuatorConnected_){
			pitch = obDouble_.segment(0,3).cast<float>();
		}
	}

	void getYaw(Eigen::Ref<EigenVec>& yaw){
		yaw = rot_randomized_.e().row(2).transpose().cast<float>();
		if(ActuatorConnected_){
			yaw = obDouble_.segment(3,3).cast<float>();
		}
	}

	void getAngularVel(Eigen::Ref<EigenVec>& angVel){
		angVel = bodyAngularVel_.cast<float>();
		if(ActuatorConnected_){
			angVel = obDouble_.segment(6,3).cast<float>();
		}
	}
	
	void getCurrentAction(Eigen::Ref<EigenVec>& currAct){
		currAct = current_action_.cast<float>();
	}
	

	double findInitializationIndex(const std::vector<double>& stdvec, const double elemToFind){
		double nearestElement = stdvec[0]; // Initialize with the first element
		double minDiff = 100;
		for (double element : stdvec) {
			double diff = std::abs(element - elemToFind);
			if (diff < minDiff) {
				minDiff = diff;
				nearestElement = element;
			}
		}
		auto it = std::find(stdvec.begin(), stdvec.end(), nearestElement); //find the value of the element
		
		if (it != stdvec.end()) 
			return  std::distance(stdvec.begin(), it);
		else
			return -1;
	}


	void select_terrain_from_tester(float stepHeight){
		select_terrain_from_tester_ = "stairs";
		stepHeight_ = stepHeight;
	}

	//do not put the height map in the reset function, otherwise each time it falls create a new environment
	void select_heightMap(){
		bool stair = rn_.intRand(0,1);
		
		if((num_episode_ >500 && stair) || (boost::iequals(select_terrain_from_tester_, "stairs"))) {
			terrainType_ = TerrainType::SingleStep;
			RSINFO_IF(visualizable_, "stairs")
			Singlestep(rn_.sampleUniform01()*stepHeight);
		}
		else{
			terrainType_ = TerrainType::Flat;
		}

	}
	void reset() final { 
		
		setFriction("ground");
		previous_height_ = 0;
		alfa_z_ = 0;
		max_height_ = 0;
		
		if((fallen_ && rn_.intRand(0,3) == 2) || rn_.intRand(0,9) == 1){
			index_imitation_ = traj_size*rn_.sampleUniform01();

			gc_init_[7] = m1_pos_(index_imitation_ - 1);
			gc_init_[8] = m2_pos_(index_imitation_ - 1);
			gc_init_[9] = 0.7*rn_.sampleUniform();
		}


		prisma_walker->setState(gc_init_, gv_init_);
		updateObservation();
		computedTorques.setZero();

		prisma_walker->getFramePosition(foot_center_, prisma_walker_pos);
	}
	
 
	float step(const Eigen::Ref<EigenVec>& action) final {
		/// action scaling

		pTarget3_ = action.cast<double>(); //dim=n_joints
		pTarget3_ = pTarget3_.cwiseProduct(actionStd_);
		actionMean_ << 0, 0, 0;
		pTarget3_ += actionMean_;
		current_action_ = pTarget3_;

		//std::cout<<"index imitation: "<<index_imitation_<<std::endl;


		motorTorque = 8*(pTarget3_ - gc_.tail(3)) - 0.8*gv_.tail(3);

		linkTorque_.tail(3) = motorTorque;
		computedTorques = prisma_walker->getGeneralizedForce().e().tail(3);


		/*if(!sea_included_){

			motorTorque = 8*(pTarget3_ - gc_.tail(3)) - 0.8*gv_.tail(3);

			for(int i = 0; i< motorTorque.size(); i++){
				motorTorque(i) = std::clamp(motorTorque[i], -gearRatio*7, gearRatio*7);
			}

			linkTorque_.tail(3) = motorTorque;
			computedTorques = prisma_walker->getGeneralizedForce().e().tail(3);

		}
		else{
			motorTorque = 2*(pTarget3_ - theta) - 0.2*dotTheta;

			for(int i = 0; i< motorTorque.size(); i++){
				motorTorque(i) = std::clamp(motorTorque[i], -gearRatio*7, gearRatio*7);
			}
			
			dotTheta += control_dt_*B_inverse*(motorTorque - motorSpring_*(theta - gc_.tail(3)));
			theta += control_dt_*dotTheta;
		
			theta(0) = std::clamp(theta(0), -1.57, 1.57);
			theta(1) = std::clamp(theta(1), -1.7, 1.57);

			if(dotTheta.squaredNorm() == 0){
				nonLinearTerms_vecDyn_ = prisma_walker->getNonlinearities(world_->getGravity());
				nonLinearTerms = nonLinearTerms_vecDyn_.e().tail(3);
				theta = gc_.tail(3) + (1/motorSpring_)*nonLinearTerms;
			}

			theta(2) = gc_(9);
			dotTheta(2) = gv_(8);

			linkTorque_.tail(3) << motorSpring_*(theta - gc_.tail(3));

			if(theta(0) >= 1.57 || theta(0) <= -1.57 || theta(1) >= 1.57 || theta(1) <= -1.57)
				rewards_.record("Joint_positions", -1);
		}*/
	
		if(implicitIntegration){
			/*Try sinuois for motors
			pTarget_.tail(nJoints_) << std::sin(2*M_PI*0.1*t), std::sin(2*M_PI*0.1*t + 0.25), 0;
			t+=control_dt_;
			*/
			/*Try open loop reference:
			pTarget_.tail(3) << m1_pos_(mettiindiceIolotolgoPerhcé i commenti sono cosi), m2_pos_(pure), 0;
			*/
			pTarget_.tail(nJoints_) << pTarget3_;
			prisma_walker->setPdTarget(pTarget_, vTarget_); //the inpit dimension of setPdTarget must be the DOF
		}else{
			prisma_walker->setGeneralizedForce(linkTorque_);
		}

		if(ActuatorConnected_){
			bool commandInPosition = true;
			if(commandInPosition){
				motors->sendCommand(pTarget3_, true);
			}
			else
				motors->sendCommand(linkTorque_.tail(nJoints_), false);
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
		
 
		ga_ = (joint_history_vel_.segment(3,3) - joint_history_vel_.segment(0,3))/control_dt_;

		rewards_.record("torque", prisma_walker->getGeneralizedForce().squaredNorm());
		rewards_.record("error_penalty", error_penalty_);
		rewards_.record("imitation", curr_imitation_*std::exp(-2*(1/sigma_square)*(error_m1_*error_m1_ + error_m2_*error_m2_)));		

		rewards_.record("velocityReward", std::exp(-2*(1/(0.3*0.3))*velocityReward_foot_or_body_));
		rewards_.record("footSlip", footSlip_);
		rewards_.record("ground_clearance", clearance_foot_);

		if(cF_["center_foot"] == 0){
			rewards_.record("BodyMovementWithLateralFeet", bodyLinearVel_.squaredNorm() + bodyAngularVel_.squaredNorm());
			//rewards_.record("StretchedLeg", (gc_[8]*gc_[8]));
		}

		actual_step_++;	
		if(actual_step_ == num_step){
			gc_init_[7] = gc_[7];
			gc_init_[8] = gc_[8];
			gc_init_[9] = gc_[9];
			gc_init_[0] = gc_[0];
			gc_init_[1] = gc_[1];
			gc_init_[2] = gc_[2];
		}
		//std::cout<<vel_foot_term_<<std::endl;

		return rewards_.sum();
		
	}

	void incrementIndices(){

		index_imitation_ += 2;
		if(index_imitation_ >= traj_size){		
			index_imitation_ = 0;
		}
	}
	
 
	void imitation_function(){
		 
		error_m1_ = gc_[7] - m1_pos_(index_imitation_);
		error_m2_ = gc_[8] - m2_pos_(index_imitation_);
		
	}

	void setPoint(){
		error_m1_ = gc_[7] - m1_pos_(setPointIndex_);
		error_m2_ = gc_[8] - m2_pos_(setPointIndex_);
		
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
		
		prisma_walker->getFrameVelocity(foot_center_, Footvel_);

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
			num_body_in_contact_ = contact.getPairObjectIndex();    
			//std::cout<<world_->getObject(contact.getPairObjectIndex())<<std::endl; 
		}


		Eigen::Vector3d x_axis;
		x_axis << cos(gc_[9]), sin(gc_[9]), 0;
		Eigen::Vector3d x_vec = rot_.e().row(0).transpose(); //x axis orientation in the world frame
		//std::cout<<"angle between the orientation of the x axis of the base and the x axis of the foot: "<<orientationDisplacement<<std::endl;
		double orientationDisplacement = acos(x_vec.dot(x_axis));
		orientationDisplacement = (orientationDisplacement*180)/M_PI;

		//Slippage
		footSlip_ = 0;
		double footVelocity_x = Footvel_[0]*Footvel_[0];
		double footVelocity_y = Footvel_[1]*Footvel_[1];
		if(cF_["center_foot"] == 1){
			footSlip_ = footVelocity_x + footVelocity_y;
			previous_height_ = 0;
			velocityReward_foot_or_body_ = std::abs(bodyAngularVel_[2] - angular_command_);
		}
		else{
			velocityReward_foot_or_body_ = std::abs(Footvel_[0] - footVelocityRef_);
		}

		
		//clearance
		if(bodyFootPos_[0] >= projectedCenterOfMass){
			curr_imitation_ = 3;
			if(previous_height_ > footPosition_[2]){ //rising phase
				clearance_foot_ = std::abs(previous_height_);	
			}
			else{
				previous_height_ = footPosition_[2];
			}
		}
		else{
			curr_imitation_ = 0.5;
		} 

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
		
 		if(ActuatorConnected_){ 	//If motors are not connected you get a seg fault om the sendRequest
			//Eigen::VectorXd obs_motors = motors->getFeedback(false, m1_pos_(index_imitation_), m2_pos_(index_imitation_));
 			//obDouble_ << obs_motors;
		}else{
			prisma_walker->getState(gc_, gv_);//generalized coordinate generalized velocity wrt initial coordinate gc_init
			
			{//clean measures to construct bodyFootPos
				quat_[0] = gc_[3]; 
				quat_[1] = gc_[4]; 
				quat_[2] = gc_[5]; 
				quat_[3] = gc_[6];

				raisim::quatToRotMat(quat_, rot_); 
				bodyPos_ = rot_.e().transpose() * gc_.segment(0, 3);  //position of the robot reported to the body frame
				prisma_walker->getFramePosition(foot_center_, footPosition_); //getFramePosition return the position w.r the world frame
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

			if(use_privileged_)
				obDouble_ << rot_randomized_.e().row(1).transpose(),
				rot_randomized_.e().row(2).transpose(), /// body orientation e' chiaro che per la camminata, e' rilevante sapere come sono orientato rispetto all'azze z, e non a tutti e 3. L'orientamento rispetto a x e' quanto sono chinato in avanti, ma io quando cammino scelgo dove andare rispetto a quanto sono chinato??? ASSOLUTAMENTE NO! Anzi, cerco di non essere chinato. Figurati un orentamento rispetto a y, significherebbe fare la ruota sul posto /// body linear&angular velocity
				bodyAngularVel_,
				joint_history_pos_, /// joint angles
				joint_history_vel_,
				current_action_,
				angular_command_;
			else
				obDouble_<<gc_.tail(3),
					gv_.tail(3),
					current_action_,
					bodyAngularVel_;
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

		joint_history_pos_.head(historyPosLength_ - numJointsControlled) = joint_history_pos_.tail(historyPosLength_ - numJointsControlled);
		joint_history_vel_.head(historyVelLength_ - numJointsControlled) = joint_history_vel_.tail(historyVelLength_ - numJointsControlled);
		 
		for(int i = 0; i < gcDim_; i++)
			gc_noise_[i] = gc_[i] + currRand_jointPos_[i]*rn_.sampleUniform(); 

		for(int i = 0; i < gvDim_; i++)
			gv_noise_[i] = gv_[i] + currRand_jointVel_[i]*rn_.sampleUniform();	

		joint_history_pos_.tail(numJointsControlled) = gc_noise_.tail(3).head(numJointsControlled);
		joint_history_vel_.tail(numJointsControlled) = gv_noise_.tail(3).head(numJointsControlled);

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
	}


	bool isTerminalState(float& terminalReward) final {
		
		if(ActuatorConnected_)
			return false;

		terminalReward = float(terminalRewardCoeff_);
		
		Eigen::Vector3d z_vec = Eigen::Vector3d::Zero();
		Eigen::Vector3d z_axis(0,0,1);
		z_vec = rot_.e().row(2).transpose();
		alfa_z_ = acos(z_vec.dot(z_axis));
		alfa_z_ = (alfa_z_*180)/M_PI;
		 
		if (std::sqrt(error_m1_*error_m1_ + error_m2_*error_m2_) > 3*sigma || alfa_z_>20){
			fallCount_++;
			index_imitation_ -= 100;
			if(index_imitation_ < 0){
				index_imitation_ = std::abs(index_imitation_ - traj_size);
			}
			gc_init_[7] = m1_pos_(index_imitation_);
			gc_init_[8] = m2_pos_(index_imitation_);
			gc_init_[9] = 0.7*rn_.sampleUniform();
			error_penalty_ += 0.4;
			fallen_ = true;
			return true;
		}		
	
		terminalReward = 0.f;
		return false;
	}

void curriculumUpdate() {
		
		angular_command_ = rn_.sampleUniform()*0.2;
		footVelocityRef_ = rn_.sampleUniform()*0.5;
		indexIncrement_ = rn_.intRand(1,10);
		setPointIndex_ = traj_size*rn_.sampleUniform01();
		
		RSINFO_IF(visualizable_, "angular velocity reference: "<<angular_command_);
		RSINFO_IF(visualizable_, "Linear foot velocity reference: "<<footVelocityRef_);

		if(terrainType_ == TerrainType::SingleStep)
			world_->removeObject(terrain_);

		actual_step_ = 0;
		error_penalty_ = 0;

		if(num_episode_ > 100){
			if(!fallen_){
				if(disturbanceGone_ = true){
					disturbanceFactor_ += 0.0025;
					disturbanceGone_ = false;
				}

				for(int i = 0; i < gcDim_; i++){
					currRand_jointPos_[i] += 0.001;
				}

				for(int i = 0; i < gvDim_; i++){
					currRand_jointVel_[i] += 0.001;
				}
				if(num_body_in_contact_ == 2){
					stepHeight = std::pow(stepHeight, curriculumDecayFactor_);
					curriculumDecayFactor_ = std::clamp(curriculumDecayFactor_ - 0.00025, 0.98, 0.99);
				}
			}
			else{
				if(num_body_in_contact_ == 2){
					stepHeight = 0.992*std::pow(10, 1/curriculumDecayFactor_*log10(stepHeight));
					curriculumDecayFactor_ = std::clamp(curriculumDecayFactor_ + 0.00025, 0.98, 0.99);
				}
				reduceRand();
			}
			RSINFO_IF(visualizable_, "stepHeight = "<<stepHeight)
			currRand_jointPos_ = currRand_jointPos_.cwiseMax(Eigen::VectorXd::Zero(gcDim_));
			currRand_jointPos_ = currRand_jointPos_.cwiseMin(Eigen::VectorXd::Constant(gcDim_, 0.5));
			currRand_jointVel_ = currRand_jointVel_.cwiseMax(Eigen::VectorXd::Zero(gvDim_));
			currRand_jointVel_ = currRand_jointVel_.cwiseMin(Eigen::VectorXd::Constant(gvDim_, 0.5));
			//RSINFO_IF(visualizable_, "current noise level" currRand_jointVel_)
		}

		if(fallen_){
			RSINFO_IF(visualizable_, "fall count = " << fallCount_)
		}
		
		fallen_ = false;
		fallCount_ = 0;
		num_episode_++;
	};


	void setFriction(std::string terrainName){
		double friction_coefficient = 0;
		if(num_episode_ < 100){   //if else are better for boolean. WHen you have nested conditions like this use switch
			friction_coefficient = 0.9;
			prisma_walker->getMass()[0] = realMass_[0] + 0.07*rn_.sampleUniform() + 0.2; 
			prisma_walker->getMass()[1] = realMass_[1] + 0.03*rn_.sampleUniform() + 0.2; 
			prisma_walker->getMass()[2] = realMass_[2] + 0.02*rn_.sampleUniform() + 0.08; 
			prisma_walker->getMass()[3] = realMass_[3] + 0.04*rn_.sampleUniform();
		}
		else if (num_episode_ < 200){
			friction_coefficient = 0.75 + rn_.sampleUniform01() * 0.15; //linear interpolation
			prisma_walker->getMass()[0] = realMass_[0] + 0.18*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[1] = realMass_[1] + 0.08*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.03*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.09*rn_.sampleUniform();
		}
		else if (num_episode_ < 500){
			friction_coefficient = 0.5 + rn_.sampleUniform01() * 0.4;
			prisma_walker->getMass()[0] = realMass_[0] + 0.30*rn_.sampleUniform() + 0.2; 
			prisma_walker->getMass()[1] = realMass_[1] + 0.12*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.04*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.15*rn_.sampleUniform();
		}
		else if (num_episode_ < 600){
			friction_coefficient = 0.45 + rn_.sampleUniform01() * 0.45;
			prisma_walker->getMass()[0] = realMass_[0] + 0.35*rn_.sampleUniform() + 0.2; 
			prisma_walker->getMass()[1] = realMass_[1] + 0.15*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.05*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.18*rn_.sampleUniform();
		}
		else if (num_episode_ <= 800){
			friction_coefficient = 0.4 + rn_.sampleUniform01() * 0.5; //[0,25,1]
			prisma_walker->getMass()[0] = realMass_[0] + 0.40*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[1] = realMass_[1] + 0.18*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.07*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.20*rn_.sampleUniform();
		}
		else if (num_episode_ > 800){
			friction_coefficient = 0.4 + rn_.sampleUniform01() * 0.5; //[0,25,1]
			prisma_walker->getMass()[0] = realMass_[0] + 0.45*rn_.sampleUniform() + 0.2; 
			prisma_walker->getMass()[1] = realMass_[1] + 0.20*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.08*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.22*rn_.sampleUniform();
		}
		/*0.951376
		0.241507
		0.1175
		0.284909*/

		prisma_walker -> updateMassInfo();
		world_->setMaterialPairProp(terrainName, "foot", friction_coefficient, 0.0, 0.001); //mu, bouayancy, restitution velocity (and minimum impact velocity to make the object bounce) 
		
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

	void Singlestep(float single_height){
		
		if(boost::iequals(select_terrain_from_tester_, "stairs")){
			single_height = stepHeight_;

		}
		float pixelSize_ = 0.02;
		terrainProp_.xSize = 2.0;
		terrainProp_.ySize = 2.0;
		terrainProp_.xSamples = terrainProp_.xSize / pixelSize_;
		terrainProp_.ySamples = terrainProp_.ySize / pixelSize_;

		heights_.resize(terrainProp_.xSamples * terrainProp_.ySamples);
		Eigen::Map<Eigen::Matrix<double, -1, -1>> mapMat(heights_.data(),
														terrainProp_.xSamples,
														terrainProp_.ySamples);
		
		double stepwidth = 0.2;
		double stepsize = stepwidth/pixelSize_;
		mapMat.setConstant(single_height);
		mapMat.block(stepwidth/pixelSize_, stepwidth/pixelSize_, terrainProp_.xSamples - 2*stepsize , terrainProp_.ySamples - 2*stepsize).setConstant(2*single_height); 
		mapMat.col(mapMat.cols() - 1).setConstant(0);
		mapMat.row(0).setConstant(0);

      	terrain_ = world_->addHeightMap(terrainProp_.xSamples,
                                    terrainProp_.ySamples,
                                    terrainProp_.xSize,
                                    terrainProp_.ySize,
                                    prisma_walker_pos[0] + 1.1, 0.0, heights_, "ground_heightMap");  //ground is the name of the material, useful to set the friction
		setFriction("ground_heightMap");
	}

	private:
		std::string home_path_;
		int gcDim_, gvDim_, nJoints_;
		float alfa_z_, roll_init_, pitch_init_, yaw_init_; // initial orientation of prisma prisma walker
		Eigen::VectorXd m1_pos_, m1_vel_;
		Eigen::VectorXd m2_pos_, m2_vel_;
 
		raisim::Mat<3,3> rot_randomized_;
 		raisim::Mat<3,3> footOrientation_; 
		Eigen::VectorXd filtered_acc_ = Eigen::VectorXd::Zero(3);
 
 		Eigen::Quaternionf q_;

		bool visualizable_ = false;
		raisim::ArticulatedSystem* prisma_walker;
		Eigen::VectorXd gc_init_, gv_init_, gc_noise_, gv_noise_, pTarget3_, vTarget_;
		const int terminalRewardCoeff_ = -12.;
		Eigen::VectorXd actionMean_, actionStd_, obDouble_;
		Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, bodyAngularVel_real_, bodyFootPos_, bodyPos_;
 
		raisim::Vec<4> quat_;
		raisim::Mat<3,3> rot_;
		size_t foot_center_;
		raisim::Vec<3> prisma_walker_pos;
		raisim::Vec<3> Footvel_, footPosition_;

		/// these variables are not in use. They are placed to show you how to create a random number sampler.
		std::normal_distribution<double> normDist_;
		RandomNumberGenerator<float> rn_;
		Actuators *motors;
		std::shared_ptr<Actuators> hebiMotor_;

		double vel_foot_term_;
		double previous_height_, max_height_, clearance_foot_ = 0;
		bool max_clearence_ = false;
		double angular_command_ = 0, footVelocityRef_ = 0;
		std::string select_terrain_from_tester_ = "niente";

		std::chrono::duration<double, std::milli> swing_time_; 
		std::chrono::steady_clock::time_point lift_instant_, land_instant_;
		std::map<std::string,int> cF_ = {
			{"center_foot", 0},
			{"lateral_feet", 0},
		};
		int num_seq, num_seq_vel, num_step;
		Eigen::VectorXd joint_history_pos_, joint_history_vel_, current_action_;
		Eigen::VectorXd joint_history_pos_reshaped_, joint_history_vel_reshaped_;
		int num_episode_ = 0;
		
		double curr_imitation_, motorSpring_; 
		bool fallen_ = false;
		int actual_step_;
		double error_m1_ = 0, error_m2_ = 0, error_m1_vel_ = 0, error_m2_vel_ = 0, error_m1_obs_ = 0, error_m2_obs_ = 0;
		double velocityReward_foot_or_body_ = 0;

		const float sigma = 0.6;
		float sigma_square = sigma*sigma;
		Eigen::VectorXd nextMotorPositions_;
		const int traj_size = 1818;
		bool ActuatorConnected_ = false;
		int curr_clearance_ = 1;
		float error_penalty_ = 0;
		int projectedCenterOfMass = 0;
		float curr_fact_slip_ = 1;

		std::vector<double> realMass_;
		Eigen::VectorXd currRand_jointPos_;
		Eigen::VectorXd currRand_jointVel_;
		float disturbanceFactor_;
		std::ofstream torques;
		int fallCount_ = 0;
		bool disturbanceGone_ = false;

		Eigen::VectorXd linkTorque_;
		Eigen::Matrix3d B_inverse;
		bool sea_included_;
		const double gearRatio = 762.222; //std::clamp wnats all the lement of the same type
		Eigen::Vector3d nonLinearTerms;
		VecDyn nonLinearTerms_vecDyn_;

		bool use_privileged_, isTerrainStairs_= false;

		int historyPosLength_;
		int historyVelLength_;

		bool implicitIntegration = false;
		bool first_time_ = false;

		float t = 0.0;
		float stepHeight_ = 0.0;
		std::vector<double> m1_stdvector_;
		std::vector<double> m2_stdvector_;
		int num_body_in_contact_ = 0;
		int numJointsControlled = 0;
		double footOrientationPenalty_;
		double footSlip_ = 0;

		//step:
		std::vector<double> heights_;
		raisim::HeightMap *terrain_;
        raisim::TerrainProperties terrainProp_;
		float stepHeight = 0.001;
		double curriculumDecayFactor_ = 0.98;
	    raisim::TerrainType terrainType_;     	       //TerrainType is not a class, but an enumerator
		int indexIncrement_ = 0;
		int setPointIndex_ = 0;

	protected:
		Eigen::VectorXd gc_, gv_, ga_, pTarget_;
		Eigen::Vector3d computedTorques;
		Eigen::Vector3d theta, dotTheta, motorTorque;
		int index_imitation_;



};
//thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}



