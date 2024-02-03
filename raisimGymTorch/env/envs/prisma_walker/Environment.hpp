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
		world_->addGround();
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
		index_imitation_[0] = 0;
		index_imitation_[1] = 0;
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
		NumberIteration_ = 0;
		/// Reward coefficients
		rewards_.initializeFromConfigurationFile (cfg["reward"]);

		footIndex_ = prisma_walker->getBodyIdx("link_3");
		
		foot_center_ = prisma_walker->getFrameIdxByLinkName("piede_interno"); //since there is a fixed joint, I cant assign a body idx to it
		foot_sx_ = prisma_walker->getFrameIdxByLinkName("piede_sx");
		foot_dx_ = prisma_walker->getFrameIdxByLinkName("piede_dx");

		m1_pos_.setZero(925);
		m2_pos_.setZero(1121); 
		/// visualize if it is the first environment
		if (visualizable_) {
			server_ = std::make_unique<raisim::RaisimServer>(world_.get());
			server_->launchServer();
			server_->focusOn(prisma_walker);
		}

		num_episode_ = 0;
		curr_vel_ = 0;
		curr_imitation_ = 1e-6;
		keyPoint_(10);
		curr_index_ = 0;
		curr_tolerance_ = 10;
		 
		motors.initHandlersAndGroup(ActuatorConnected_, num_seq, num_seq_vel, visualizable_);

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

	void reset() final {
		
		previous_height_ = 0;
		alfa_z_ = 0;
		max_height_ = 0;
		clearance_foot_ = 0.1; //penalita' iniziale. Altrimenti non alzera' mai il piede

		std::ifstream m1_traj;  //fstream is for both input and output, but hen you need to specify with std::ios::in
    	std::ifstream m2_traj;

		m1_traj.open("m1.txt", std::ios::in);
    	m2_traj.open("m2.txt", std::ios::in);
	
		Eigen::VectorXd m1_pos(num_row_in_file);
		Eigen::VectorXd m2_pos(num_row_in_file);

		if(m1_traj.is_open() && m2_traj.is_open()){

			for(int j = 0; j<m1_pos_.size() ;j++)
				m1_traj >> m1_pos_(j);  //one character at time store the content of the file inside the vector
				//m1_pos_=-m1_pos;
			for(int j = 0; j<m2_pos_.size() ;j++)
				m2_traj >> m2_pos_(j);
				//m2_pos_=-m2_pos;
	
		}
		
		m1_traj.close(); 
		m2_traj.close();
		previous_contact = 0;


		if(fallen_){
			index_imitation_[0] = 1818*rn_.sampleUniform01();
			index_imitation_[1] = 1818*rn_.sampleUniform01();

			gc_init_[7] = m1_pos_(index_imitation_[0] - 1);
			gc_init_[8] = m2_pos_(index_imitation_[1] - 1);
		}
	

		prisma_walker->setState(gc_init_, gv_init_);
		updateObservation();
	
		NumberIteration_ = 0;
		offset_ = 1;
		//Curriculum
		
		countForCurriculum_ = 0;

		error_penalty_ = 0;
		swing_penalty_ = 0; //Da usare solo in caso di penalty
		previous_contact = 1; //Parte da piede a terra e quindi va a 1
		actual_step_ = 0;
	}
	
 
	float step(const Eigen::Ref<EigenVec>& action) final {
		/// action scaling
		pTarget3_ = action.cast<double>(); //dim=n_joints
		pTarget3_ = pTarget3_.cwiseProduct(actionStd_);

		actionMean_ << m1_pos_(index_imitation_[0]), m2_pos_(index_imitation_[1]), 0.0;
		pTarget3_ += actionMean_;
		pTarget_.tail(nJoints_) << pTarget3_;
		current_action_ = pTarget3_;

	    if(ActuatorConnected_){
			motors.sendCommand(pTarget3_);
		}
		else{
			prisma_walker->setPdTarget(pTarget_, vTarget_);
		}


		for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
			if(server_) server_->lockVisualizationServerMutex();
			world_->integrate();
			if(server_) server_->unlockVisualizationServerMutex();
		}

		updateObservation(); //Ricordati che va prima di fare tutti i conti

		contacts(); //va dopo il world integrate, perche' il world integrate aggiorna i contatti. 
 
		rewards_.record("torque", prisma_walker->getGeneralizedForce().squaredNorm());
		rewards_.record("Joint_velocity", gv_.segment(6,2).squaredNorm());
		rewards_.record("Joint_velocity", 5*gv_.tail(1).squaredNorm());

		rewards_.record("error_penalty", error_penalty_);

		rewards_.record("imitation", imitation_function());
		rewards_.record("dynamixel_joint", std::exp(-2*(1/sigma)*gc_[9]*gc_[9]));
		rewards_.record("angular_penalty", 0.05*(bodyAngularVel_[0] + bodyLinearVel_[1]));
		rewards_.record("slip", slip_term_);
		rewards_.record("standing_still", standingStill_);

		actual_step_++;	
		if(actual_step_ == num_step){
			gc_init_[7] = gc_[7];
			gc_init_[8] = gc_[8];
		}
 
		return rewards_.sum();
		
	}

	float SwingPenalty(){
		if(swing_time_ > std::chrono::duration<double, std::milli>(1) ){
			swing_penalty_ += 0.01;	
		}
		else{
			swing_penalty_ = 0;
		}
		return swing_penalty_;

	}

	float imitation_function (){
		
	 
		index_imitation_[0] += 1;
		index_imitation_[1] += 1;

		if(index_imitation_[0] > 925)
			index_imitation_[0] = 0;

		if(index_imitation_[1] > 1121)
			index_imitation_[1] = 0;

		
		
		error_m1_ = gc_[7] - m1_pos_(index_imitation_[0]);
		error_m2_ = gc_[8] - m2_pos_(index_imitation_[1]);
		double gaussian_kernel = std::exp(-2*(1/sigma_square)*(error_m1_*error_m1_ + error_m2_*error_m2_));
		
		return gaussian_kernel;

	}
 
	float norm(Eigen::Vector3d vector){
		float norma=0;
		norma = sqrt(pow(vector(0),2)+pow(vector(1),2)+pow(vector(2),2));

		return norma;
	}


	void clearance(){
		if(cF_["center_foot"]==0){  //Swing phase
			if(previous_height_ > footPosition_[2]){// descent phase
				if(max_clearence_){ //during all the descendant phase, the agent receive the same penalty because wasnt able to lift enough the foot
					clearance_foot_ = previous_height_;
					max_clearence_ = false;
					//RSINFO_IF(visualizable_, "clearance: ", clearance_foot_)
				}
			}else{  //rising phase 
				previous_height_ = footPosition_[2];    
				max_clearence_ = true;
			}	
		}
		else
			previous_height_ = 0;

	}

	void slippage(){
		slip_term_ = 0;
		if(cF_["center_foot"] == 1 || footPosition_[2] < 0.001){
			slip_term_ += std::sqrt(vel_[0]*vel_[0] + vel_[1]*vel_[1]);
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

	
	void error_vel(){
		if(cF_["center_foot"]==1){
			double err_x = command_[0] - bodyLinearVel_[0];
			double err_omega = command_[2] - bodyAngularVel_[2];
			lin_reward_ =  std::exp(-5*err_x*err_x);
			ang_reward_ =  std::exp(-15*err_omega*err_omega);
			previous_reward_ = lin_reward_ + ang_reward_;
		}else{		
			if(swing_time_ >= std::chrono::duration<double, std::milli>(1500) ){
				lin_reward_ += -0.1;
				ang_reward_ += -0.1;
			}
		}
	}

	void mean_square_error(){
		double err_x = command_[0] - bodyLinearVel_[0];
		double err_omega = command_[2] - bodyAngularVel_[2];
		vel_rew_ += (err_x*err_x) + (err_omega*err_omega);   //se la fase di swing dura poco fa poche iterazioni!!!!!!
	}

	void contacts(){
		
		prisma_walker->getFramePosition(foot_center_, footPosition_);
	
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

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		if(cF_["center_foot"] == 1 && cF_["lateral_feet"] == 1){
			elapsed_time_ += std::chrono::duration<double, std::milli>(end - start_); 
			start_ = std::chrono::steady_clock::now();
			if(elapsed_time_ > std::chrono::duration<double, std::milli>(1000))
				standingStill_ += 1;
		}
		else{
			standingStill_ = 0;
			elapsed_time_ = std::chrono::duration<double, std::milli>(0);
			start_ =  std::chrono::steady_clock::now();
		}

		slippage();
		clearance();
		//mse_and_airTime();
	} 
 

	inline void mse_and_airTime(){

		if(cF_["center_foot"] == 1){  //NON TENERE LE COSE DENTRO IL FOR, PERCHè Altrimenti chiama le stesse funzioni piu' VOLTE!!!
			if(previous_contact == 0 && cF_["center_foot"]==1){ //piede atterra
				swing_time_ = std::chrono::duration<double, std::milli>(0);
			}
			lift_instant_ = std::chrono::steady_clock::now();
			//mean_square_error();
			NumberIteration_ ++;
		}

		previous_reward_ = 0;
	
		if(cF_["center_foot"] == 0){
			land_instant_ = std::chrono::steady_clock::now();
			swing_time_ += std::chrono::duration<double, std::milli>(land_instant_ - lift_instant_);
			lift_instant_ = std::chrono::steady_clock::now();

			/*if(previous_contact == 1 && cF_["center_foot"]==0){// Inizia a sollevare-> Robot Fermo
				bool start = true;
				
				/*mean_vel_rew_ = rew_x_ / NumberIteration_; //Riempe previous vel con l'errore medio
				NumberIteration_ = 1;
				vel_rew_ = 0;
			}		*/
		}

		previous_contact = cF_["center_foot"];		

		//RSINFO_IF(visualizable_, "swing time: "<< swing_time_.count())
		//RSINFO_IF(visualizable_, "error vel: "<< mean_rew_x_)
		swing_time_d = swing_time_.count()/1000;
 
	}



	void generate_command_velocity(){ 
		command_[0] = 0;
		command_[2] = 0;
	}
		

	void updateObservation() {
		
		nextMotorPositions_ << m1_pos_(index_imitation_[0]), m1_pos_(index_imitation_[0] + 1), m1_pos_(index_imitation_[0] +2), m2_pos_(index_imitation_[1]), m2_pos_(index_imitation_[1] + 1), m2_pos_(index_imitation_[1] +2);
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
			bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3); //linear velocity reported to the base frame
			bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);			

			obDouble_ << rot_.e().row(1).transpose(),
			rot_.e().row(2).transpose(), /// body orientation e' chiaro che per la camminata, e' rilevante sapere come sono orientato rispetto all'azze z, e non a tutti e 3. L'orientamento rispetto a x e' quanto sono chinato in avanti, ma io quando cammino scelgo dove andare rispetto a quanto sono chinato??? ASSOLUTAMENTE NO! Anzi, cerco di non essere chinato. Figurati un orentamento rispetto a y, significherebbe fare la ruota sul posto
			bodyAngularVel_, /// body linear&angular velocity
			joint_history_pos_, /// joint angles
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

	
		joint_history_pos_.tail(3) = gc_.tail(3);
		joint_history_vel_.tail(3) = gv_.tail(3);
		
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
		alfa_z_ = acos((z_vec.dot(z_axis))/norm(z_vec));
		alfa_z_ = (alfa_z_*180)/M_PI;
	
		//RSINFO_IF(visualizable_, alfa_z_*alfa_z_)
		if(alfa_z_>25){
			RSINFO_IF(visualizable_, "fallen for angle")
			fallen_ = true;
			return true;
		}

		if (std::sqrt(error_m1_*error_m1_ + error_m2_*error_m2_) > curr_tolerance_*sigma){
			RSINFO_IF(visualizable_, "fallen for error")
			error_penalty_ += curr_tolerance_*0.1;
			fallen_ = true;
			fallen_error = true;
			return true;
		}

		if(cF_["center_foot"] == 0 && bodyLinearVel_.squaredNorm() !=0){
			return true;
		}		

		terminalReward = 0.f;
		return false;
	}

	void curriculumUpdate() {
		//generate_command_velocity(); //La metto qui perche' la reset viene chiamata troppe volte

		if(num_episode_ > 10 && !fallen_error){
			curr_tolerance_ += -0.05;
			curr_imitation_ += 1e-7; 
		}

		if(fallen_error){
			curr_tolerance_ += 0.25;
			curr_imitation_ -= 2e-7;
		}

		if(curr_tolerance_ < 2)
			curr_tolerance_ = 2;

		if(curr_imitation_ > 1e-4)
			curr_imitation_ = 1e-4;

		num_episode_++;

		if(visualizable_)
			std::cout<<"Fallen: "<<fallen_error<<"   Curr tolerance: "<<curr_tolerance_<<std::endl;

		//IT must go at the end of the episode
		fallen_ = false;
		fallen_error = false;

	};



	private:
		std::string home_path_;
		int gcDim_, gvDim_, nJoints_,timing_,fbk_counter_,n_campione_vel_,n_campione_pos_;
		int index_imitation_[2];
		float alfa_z_, roll_init_, pitch_init_, yaw_init_; // initial orientation of prisma prisma walker

		Eigen::VectorXd m1_pos_;
		Eigen::VectorXd m2_pos_;
 
		double smoothing_factor_ = 0.06;
		raisim::Mat<3,3> rot_off_;
		raisim::Vec<4> quaternion_;
 
		Eigen::VectorXd filtered_acc_ = Eigen::VectorXd::Zero(3);
 
 		Eigen::Quaternionf q_;
	 	double error_penalty_;
		double mean_value_x_ = 0.0;double mean_value_y_ = 0.0;double mean_value_z_ = 0.0;
		Eigen::VectorXd real_lin_acc_ = Eigen::VectorXd::Zero(3);

		Eigen::Matrix3d R_imu_base_frame_ = Eigen::Matrix3d::Zero(3,3);
		Eigen::Matrix3d real_orientation_matrix_ = Eigen::Matrix3d::Zero(3,3);
		Eigen::Vector3d rpy_;
		bool visualizable_ = false;
		raisim::ArticulatedSystem* prisma_walker;
		Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget3_, vTarget_;
		const int terminalRewardCoeff_ = -12.;
		Eigen::VectorXd actionMean_, actionStd_, obDouble_;
		Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, bodyAngularVel_real_;
		std::ofstream v_lin_sim_ = std::ofstream("v_lin_sim_40.txt");
		std::ofstream p_sim_ = std::ofstream("p_sim_40.txt");
		std::ofstream ori_sim_ = std::ofstream("ori_sim_40.txt");
		std::ofstream v_ang_sim_ = std::ofstream("v_ang_sim_40.txt");
		raisim::Vec<4> quat_;
		raisim::Mat<3,3> rot_;
		size_t foot_center_, footIndex_, foot_sx_ ,foot_dx_, contact_;
		raisim::CoordinateFrame footframe_,frame_dx_foot_,frame_sx_foot_;
		raisim::Vec<3> vel_, ang_vel_, vel_sx_, vel_dx_, ang_vel_sx_, ang_vel_dx_, footPosition_;
		/// these variables are not in use. They are placed to show you how to create a random number sampler.
		std::normal_distribution<double> normDist_;
		RandomNumberGenerator<float> rn_;
		Actuators motors;

		double slip_term_;
		double previous_height_, max_height_, clearance_foot_ = 0;
		bool max_clearence_ = false;
		Eigen::Vector3d command_;

		std::chrono::duration<double, std::milli> elapsed_time_; 
		std::chrono::duration<double, std::milli> swing_time_; 
		std::chrono::steady_clock::time_point start_, lift_instant_, land_instant_;
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
		int NumberIteration_, countForCurriculum_;
		double curr_imitation_, curr_vel_, curr_tolerance_;
		Eigen::VectorXd keyPoint_; 
		std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> footPositions_;
		bool gonext_ = false;
		float swing_penalty_;
		int offset_, curr_index_; 
		bool fallen_, fallen_error = false;
		int actual_step_;
		double error_m1_ = 0, error_m2_ = 0;
		const float sigma = 0.23;
		float sigma_square = sigma*sigma;
		Eigen::VectorXd nextMotorPositions_;

		bool ActuatorConnected_ = false;

};
//thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

