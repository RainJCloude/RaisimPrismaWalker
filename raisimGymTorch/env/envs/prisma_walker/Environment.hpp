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
		home_path_ = "/home/dev/raisim_ws/raisimlib";
		/// add objects
		prisma_walker = world_->addArticulatedSystem(home_path_ + "/rsc/prisma_walker/urdf/prisma_walker.urdf");
		prisma_walker->setName("prisma_walker");
		prisma_walker->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
		world_->addGround(0, "ground");
		Eigen::Vector3d gravity_d(0.0,0.0,0);
		/// get robot data
		gcDim_ = prisma_walker->getGeneralizedCoordinateDim();
		gvDim_ = prisma_walker->getDOF();
		nJoints_ = gvDim_ - 6;

		/// initialize containers
		gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
		gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
		pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget3_.setZero(nJoints_);  
		roll_init_= 0.0;pitch_init_=0.0;yaw_init_=0.0;
		q_ = Eigen::AngleAxisf(roll_init_, Eigen::Vector3f::UnitX())
		* Eigen::AngleAxisf(pitch_init_, Eigen::Vector3f::UnitY())
		* Eigen::AngleAxisf(yaw_init_, Eigen::Vector3f::UnitZ());
		/// this is nominal configuration of prisma walker
		gc_init_ << 0.0, 0.0, 0.33, q_.w(), q_.x(), q_.y(), q_.z(), 0.6, 0.6, 0.0; //(pos, orientament, configurazione)

		/// set pd gains
		Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
		jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(20.0);
		jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(2.0);
		//usando un d gain 0.2 non converge a niente, mettendolo a 2 invece per un po' sembra migliorare ma poi torna a peggiorare
		int max_time;
		READ_YAML(int, num_seq, cfg_["num_seq"]);
   		READ_YAML(int, num_seq_vel, cfg_["num_seq_vel"]);
		READ_YAML(int, max_time, cfg_["max_time"]);  //mu,Step = control_dt /max time
		num_step = max_time/control_dt_;
		joint_history_pos_.setZero(3*num_seq);
    	joint_history_vel_.setZero(3*num_seq_vel);
		nextMotorPositions_.setZero(2*num_seq);

		current_action_.setZero(3);
		index_imitation_ = 0;
		prisma_walker->setPdGains(jointPgain, jointDgain);
		prisma_walker->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

		/// MUST BE DONE FOR ALL ENVIRONMENTS
		obDim_ = 11 + joint_history_pos_.size() + joint_history_vel_.size() + current_action_.size() + nextMotorPositions_.size();
		actionDim_ = nJoints_; 
		
		actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
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
		curr_vel_ = 0;
		curr_imitation_ = 1e-6;
		curr_index_ = 0;
		actual_step_ = 0;
	 	clearance_foot_ = 0.0; //penalita' iniziale. Altrimenti non alzera' mai il piede

		//motors.initHandlersAndGroup(ActuatorConnected_, num_seq, num_seq_vel, visualizable_);
		openFile();

		for(int i = 0; i < prisma_walker->getMass().size(); i++){
      		realMass_.push_back(prisma_walker->getMass()[i]);	//it has 4 links (it takes into account only the bodies with not fixed joints)
			//std::cout<<realMass_[i]<<std::endl;
			//the link 0 is the base with the fixed legs m = 0.951376
			//link 1 is that attached to the first hebi m = 0.24
			//link 2 is that attached to the second hebi m = 0.1175
			//link 3 is the foot attached to the dynamixel m = 0.286
		}
		
		prisma_walker->getCollisionBody("piede_interno/0").setMaterial("foot");

		currRand_jointPos_.setZero(gcDim_ - 3); // 7 elements
		currRand_jointVel_.setZero(gvDim_ - 3); // 6 elements
		disturbanceFactor_ = 0.5;
		/*torques.open(home_path_ + "/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/torques.txt");
		torques << "Hebi 1 torque" << "\t" << "Hebi 2 torque" << "\t" << "M3 torque" << "\n"; */
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
		fallen_ = false;
	}

	void init() final { 
		//Enable torque: activate 
		//int dxl_comm_result = packetHandler_->write1ByteTxRx(portHandler_, dxl_id, addrTorqueEnable, TorqueEnable, &dxl_error_);

	}     

	void reset() final {

		setFriction();

		previous_height_ = 0;
		alfa_z_ = 0;
		
		if((fallen_ && rn_.intRand(0,7) == 1) || rn_.intRand(0,20) == 1){
			index_imitation_ = traj_size*rn_.sampleUniform01();

			gc_init_[7] = m1_pos_(index_imitation_ - 1);
			gc_init_[8] = m2_pos_(index_imitation_ - 1);

			initBodyPos_ =  rot_.e().transpose() * gc_init_.segment(0, 3);
			initFootPos_ =  rot_.e().transpose() * footPosition_.e();
			initFootPos_ = initFootPos_ - initBodyPos_;
			
		}

		if(fallen_ && num_episode_ > 1000)
			reduceRand();


		prisma_walker->setState(gc_init_, gv_init_);
		updateObservation();

		swing_penalty_ = 0; //Da usare solo in caso di penalty
		previous_contact = 1; //Parte da piede a terra e quindi va a 1
		error_penalty_ = 0;
	}
	
 
	float step(const Eigen::Ref<EigenVec>& action) final {
		/// action scaling
		pTarget3_ = action.cast<double>(); //dim=n_joints
		pTarget3_ = pTarget3_.cwiseProduct(actionStd_);
		actionMean_ << m1_pos_(index_imitation_), m2_pos_(index_imitation_), 0.0;
		pTarget3_ += actionMean_;
		pTarget_.tail(nJoints_) << pTarget3_;
		current_action_ = pTarget3_;
	    if(ActuatorConnected_){
			motors.sendCommand(pTarget3_);
		}
		else{
			prisma_walker->setPdTarget(pTarget_, vTarget_);
		}

		/*
		Eigen::Vector3d torque;

		torque << 20*(gc_[7] - pTarget3_[0]) + 2*gv_[6],
				  20*(gc_[8] - pTarget3_[1]) + 2*gv_[7],
				  20*(gc_[9] - pTarget3_[2]) + 2*gv_[8];

		torques << torque[0] << std::setw(5) << torque[1] << std::setw(5) << torque[2] << std::setw(5);
		torques <<"\n";*/

		for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
			if(server_) server_->lockVisualizationServerMutex();
			world_->integrate();
			if(server_) server_->unlockVisualizationServerMutex();
		}

		updateObservation(); //Ricordati che va prima di fare tutti i conti
		contacts(); //va dopo il world integrate, perche' il world integrate aggiorna i contatti. 
		imitation_function();
		external_force();

		rewards_.record("torque", prisma_walker->getGeneralizedForce().squaredNorm());
		//rewards_.record("Joint_velocity", curr_fact* gv_.segment(6,2).squaredNorm());
		//rewards_.record("Joint_velocity", curr_fact*5*gv_.tail(1).squaredNorm());
		rewards_.record("error_penalty", error_penalty_);
		double errorImit = std::exp(-2*(1/sigma_square)*(error_m1_*error_m1_ + error_m2_*error_m2_));
		rewards_.record("imitation", errorImit);
		//rewards_.record("dynamixel_joint", std::exp(-2*(1/sigma)*gc_[9]*gc_[9]));
		rewards_.record("angular_penalty", bodyAngularVel_[0] + bodyAngularVel_[1]); //+0.025*bodyLinearVel_[2]
		rewards_.record("slip", curr_fact_slip_*slip_term_);
		rewards_.record("ground_clearence", clearance_foot_);
		if(visualizable_){
			if(isnan(errorImit))
				std::cout<<"Erroror imitation"<<std::endl;
			if(isnan(slip_term_))
				std::cout<<"Error slp term"<<std::endl;
			if(isnan(clearance_foot_))
				std::cout<<"Error clearance_foot_term"<<std::endl;		
			if(isnan(error_penalty_))
			std::cout<<"Error penalty term"<<std::endl;
		}
		if(cF_["center_foot"] == 0)
			rewards_.record("BodyMovementWithLateralFeet", bodyLinearVel_.squaredNorm() + bodyAngularVel_.squaredNorm());
		//rewards_.record("air_foot", SwingPenalty());

		actual_step_++;	
		if(actual_step_ == num_step){
			gc_init_[7] = gc_[7];
			gc_init_[8] = gc_[8];
		}

		//std::cout<<slip_term_<<std::endl;	

		incrementIndices();
		return rewards_.sum();
	
	}

	void incrementIndices(){

		index_imitation_++;

		if(index_imitation_ > traj_size){
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
		
		error_m1_ = gc_[7] - m1_pos_(index_imitation_);
		error_m2_ = gc_[8] - m2_pos_(index_imitation_);
	}
 

	void clearance(){
	
		if(bodyFootPos_[0] >= posForFootHitGround_ && initFootPos_[0] <= posForFootHitGround_){
			sigma_square = sigma*sigma;
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
			initFootPos_[0] = -1; //become negative the fullfill the previous condition
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
			if(previous_contact == 0 && cF_["center_foot"]==1){ //piede atterra
				swing_time_ = std::chrono::duration<double, std::milli>(0);
			}
			lift_instant_ = std::chrono::steady_clock::now();
			//mean_square_error();
		}

		previous_reward_ = 0;
	
		if(cF_["center_foot"] == 0){
			land_instant_ = std::chrono::steady_clock::now();
			swing_time_ += std::chrono::duration<double, std::milli>(land_instant_ - lift_instant_);
			lift_instant_ = std::chrono::steady_clock::now();
		}

		previous_contact = cF_["center_foot"];		

		swing_time_d = swing_time_.count()/1000;
	}


	void updateObservation() {
		
		nextMotorPositions_ << m1_pos_(index_imitation_), m1_pos_(index_imitation_ + 1), m1_pos_(index_imitation_ +2), m2_pos_(index_imitation_), m2_pos_(index_imitation_ + 1), m2_pos_(index_imitation_ +2);
		//Il giunto di base e' fisso rispetto alla base. Quindi l'orientamento della base e' quello del motore 
		
		if(ActuatorConnected_)
		{ 	//If motors are not connected you get a seg fault om the sendRequest
			Eigen::VectorXd obs_motors = motors.getFeedback();
 			obDouble_ << obs_motors,
			current_action_,
			error_m1_,
			error_m2_;
			nextMotorPositions_; 

		}else{
			prisma_walker->getState(gc_, gv_);//generalized coordinate generalized velocity wrt initial coordinate gc_init
			updateJointHistory();

			quat_[0] = gc_[3]; 
			quat_[1] = gc_[4]; 
			quat_[2] = gc_[5]; 
			quat_[3] = gc_[6];

			raisim::quatToRotMat(quat_, rot_); 

			bodyPos_ = rot_.e().transpose() * gc_.segment(0, 3);  //position of the robot reported to the body frame
			prisma_walker->getFramePosition(foot_center_, footPosition_);
			bodyFootPos_ = rot_.e().transpose() * footPosition_.e();
			bodyFootPos_ = bodyFootPos_ - bodyPos_; //otherwise it changes only when I move the foot, it must be changed also when the robot moves the body

			for(int i = 0; i < currRand_jointPos_.size(); i++)
				gc_[i+3] += currRand_jointPos_[i]*rn_.sampleUniform(); 
 
			for(int i = 0; i < currRand_jointVel_.size(); i++)
				gv_[i] += currRand_jointPos_[i]*rn_.sampleUniform();		 

			quat_[0] = gc_[3]; 
			quat_[1] = gc_[4]; 
			quat_[2] = gc_[5]; 
			quat_[3] = gc_[6];

			raisim::quatToRotMat(quat_, rot_randomized_); 

		
			bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3); //linear velocity reported to the base frame
			bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);
	
			obDouble_ << rot_.e().row(1).transpose(),
			rot_.e().row(2).transpose(), /// body orientation e' chiaro che per la camminata, e' rilevante sapere come sono orientato rispetto all'azze z, e non a tutti e 3. L'orientamento rispetto a x e' quanto sono chinato in avanti, ma io quando cammino scelgo dove andare rispetto a quanto sono chinato??? ASSOLUTAMENTE NO! Anzi, cerco di non essere chinato. Figurati un orentamento rispetto a y, significherebbe fare la ruota sul posto /// body linear&angular velocity
			bodyAngularVel_,
			joint_history_pos_, ///
			joint_history_vel_,
			current_action_,
			error_m1_,
			error_m2_,
			nextMotorPositions_; 
		}
	
	}


	void updateJointHistory(){

		Eigen::VectorXd temp_pos (3*num_seq);
		temp_pos << joint_history_pos_; //temp conterrà nelle posizioni 0-11 quello che joints_history conterra' nelle posizioni 12-23
		
		Eigen::VectorXd temp_vel (3*num_seq_vel);
		temp_vel << joint_history_vel_;

		for(int i = 0; i < (num_seq-1); i++){
			joint_history_pos_(Eigen::seq((i)*3, (i+1)*3-1)) = temp_pos(Eigen::seq((i+1)*3, (i+2)*3-1)); //overwrite the next sequence
		}

		for(int i = 0; i < (num_seq_vel-1); i++){
			joint_history_vel_(Eigen::seq((i)*3, (i+1)*3-1)) = temp_vel(Eigen::seq((i+1)*3, (i+2)*3-1));
		}

		//PER FARE DELLE PROVE, USA QUESTI VETTORI CHE HANNO SOLO NUMERI DA 1 a 11
		/*joint_history_pos_.tail(3) = Eigen::ArrayXd::LinSpaced(3, 0, 2); //genera un vettore di 12 elementi di numeri equidistanziati che vanno da 0 a 12
		joint_history_vel_.tail(3) = Eigen::ArrayXd::LinSpaced(3, 0, 2);
		if(visualizable_){
			std::cout<<"Joint pos: "<< joint_history_pos_<< std::endl;
			std::cout<<"Joint vel: "<< joint_history_vel_<< std::endl;
		}*/
		Eigen::Vector3d randVecThree;

		randVecThree << 2*rn_.sampleUniform(), 2*rn_.sampleUniform(), 2*rn_.sampleUniform();
		joint_history_pos_.tail(3) = gc_.tail(3) + randVecThree;

		randVecThree << 2*rn_.sampleUniform(), 2*rn_.sampleUniform(), 2*rn_.sampleUniform();
		joint_history_vel_.tail(3) = gv_.tail(3) + randVecThree;
		
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
	
		//RSINFO_IF(visualizable_, alfa_z_*alfa_z_)
		if(alfa_z_>18){
			fallCount_++;
			gc_init_[7] = m1_pos_(index_imitation_ - 100);
			gc_init_[8] = m2_pos_(index_imitation_ - 100);
			fallen_ = true;
			return true;
		}

	 
		if (std::sqrt(error_m1_*error_m1_ + error_m2_*error_m2_) > 6*sigma){
			fallCount_++;
			gc_init_[7] = m1_pos_(index_imitation_ - 100);
			gc_init_[8] = m2_pos_(index_imitation_ - 100);
			error_penalty_ += 0.4;
			fallen_ = true;
			return true;
		}		
	
		
		
		terminalReward = 0.f;
		return false;
	}

	void setFriction(){
		double friction_coefficient = 0;
		if(num_episode_ < 200){
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
		else if (num_episode_ <= 1100){
			friction_coefficient = 0.3 + rn_.sampleUniform01() * 0.2;
			prisma_walker->getMass()[0] = realMass_[0] + 0.40*rn_.sampleUniform() + 0.2; //add 100g of te base motor
			prisma_walker->getMass()[1] = realMass_[1] + 0.18*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.05*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.20*rn_.sampleUniform();
		}
		else if (num_episode_ > 1100){
			friction_coefficient = 0.25 + rn_.sampleUniform01() * 0.75; //[0,25,1]
			prisma_walker->getMass()[0] = realMass_[0] + 0.45*rn_.sampleUniform() + 0.2; //add 100g of te base motor
			prisma_walker->getMass()[1] = realMass_[1] + 0.20*rn_.sampleUniform() + 0.2;
			prisma_walker->getMass()[2] = realMass_[2] + 0.05*rn_.sampleUniform() + 0.08;
			prisma_walker->getMass()[3] = realMass_[3] + 0.23*rn_.sampleUniform();
		}
		
		prisma_walker -> updateMassInfo();
		world_->setMaterialPairProp("ground", "foot", friction_coefficient, 0.0, 0.001); //mu, bouayancy, restitution velocity (and minimum impact velocity to make the object bounce) 
		
	}


	void curriculumUpdate() {
		//generate_command_velocity(); //La metto qui perche' la reset viene chiamata troppe volte

		if(num_episode_ > 10 && !fallen_){
			curr_imitation_ += 1e-6; 
		}

		if(fallen_ && curr_imitation_ > 1e-6){
			curr_imitation_ -= 2e-6;
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
		fallen_ = false;
		actual_step_ = 0;

		if(num_episode_ > 300){
			if(!fallen_){
				if(disturbanceGone_ = true){
					disturbanceFactor_ += 0.5;
					disturbanceGone_ = false;
				}

				for(int i = 0; i < currRand_jointPos_.size(); i++){
					if(currRand_jointPos_[i] < 3)
						currRand_jointPos_[i] += 0.001;
				}

				for(int i = 0; i < currRand_jointVel_.size(); i++){
					if(currRand_jointVel_[i] < 3)
						currRand_jointVel_[i] += 0.001;
				}
			}
			else{
				reduceRand();
			}
		}

		/*
		if(num_episode_ % 1000 == 0){
			if(visualizable_){
				for(int i = 0; i < currRand_jointPos_.size(); i++)
					std::cout<<"Randomizing factor for joint pos: "<<currRand_jointPos_[i]<<std::endl;

				for(int i = 0; i < currRand_jointVel_.size(); i++)
					std::cout<<"Randomizing factor for joint vel: "<<currRand_jointVel_[i]<<std::endl;
			}
		}*/
		if(fallen_){
			RSINFO_IF(visualizable_, "fall count = " << fallCount_)
		}
		fallCount_ = 0;
	};

	void reduceRand(){

		if(disturbanceFactor_ > 1)
			disturbanceFactor_ -= 0.5;
		
		for(int i = 0; i < currRand_jointPos_.size(); i++)
			currRand_jointPos_[i] -= 0.001;

		for(int i = 0; i < currRand_jointVel_.size(); i++)
			currRand_jointVel_[i] -= 0.001;	

		int indexPos = rn_.intRand(0,7);
		int indexVel = rn_.intRand(0,6);

		if(currRand_jointPos_[indexPos] < 3)
			currRand_jointPos_[indexPos] += 0.001;

		if(currRand_jointPos_[indexVel] < 3)
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
			disturbance[2] = rn_.sampleUniform();
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
		Eigen::VectorXd real_lin_acc_ = Eigen::VectorXd::Zero(3);

		Eigen::Matrix3d R_imu_base_frame_ = Eigen::Matrix3d::Zero(3,3);
		Eigen::Vector3d rpy_;
		bool visualizable_ = false;
		raisim::ArticulatedSystem* prisma_walker;
		Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget3_, vTarget_;
		const int terminalRewardCoeff_ = -12.;
		Eigen::VectorXd actionMean_, actionStd_, obDouble_;
		Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, bodyAngularVel_real_, bodyFootPos_, bodyPos_, initBodyPos_, initFootPos_;
		raisim::Vec<4> quat_;
		raisim::Mat<3,3> rot_;
		size_t foot_center_, footIndex_, foot_sx_ ,foot_dx_, contact_;
		raisim::CoordinateFrame footframe_,frame_dx_foot_,frame_sx_foot_;
		raisim::Vec<3> vel_, footPosition_;
		/// these variables are not in use. They are placed to show you how to create a random number sampler.
		std::normal_distribution<double> normDist_;
		RandomNumberGenerator<float> rn_;
		Actuators motors;

		double slip_term_;
		double previous_height_, clearance_foot_ = 0;
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
		double previous_reward_, lin_reward_, ang_reward_ = 0;
		
		int previous_contact;
		int count_ = 0;
		double vel_rew_, mean_vel_rew_, swing_time_d;
		int countForCurriculum_;
		double curr_imitation_, curr_vel_;
		std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> footPositions_;
		bool gonext_ = false;
		float swing_penalty_;
		int curr_index_; 
		bool fallen_ = false;
		int actual_step_;
		double error_m1_ = 0, error_m2_ = 0;
		const float sigma = 0.23;
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

};
//thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

