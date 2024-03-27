#ifndef ACTUATORS_HPP_
#define ACTUATORS_HPP_

#include "lookup.hpp"
#include "group_command.hpp"
#include "group_feedback.hpp"
#include "command.hpp"
#include "dynamixel_sdk.h"


// Control table address
constexpr int addrTorqueEnable = 64;            // Control table address is different in Dynamixel model
constexpr int addrGoalPosition = 116;
constexpr int addrPresentPosition = 132;
constexpr int addrPresentVelocity = 128;

// Protocol version
constexpr int protocolVersion = 2.0;            // See which protocol version is used in the Dynamixel

// Default setting
constexpr int dxl_id = 1;  // Dynamixel ID: 1
constexpr int baudrate = 2000000;
#define deviceName "/dev/ttyUSB0"     // Check which port is being used on your controller

constexpr int TorqueEnable = 1;                 		// Value for enabling the torque
constexpr int TorqueDisable = 0;                  		// Value for disabling the torque
constexpr int dxlMinimumPositionValue = 0;          	// Dynamixel will rotate between this value
constexpr int dxlMaximumPositionValue = 4095;      		// and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
constexpr int dxlMovingStatusThreshold = 20;            // Dynamixel moving status threshold


class Actuators {

 public:

	Actuators(){

	};

	void initHandlersAndGroup(bool & ActuatorConnected, int num_pos, int num_vel, bool visualizable_, int controlStrategyInt)
	{			
		portHandler_ = dynamixel::PortHandler::getPortHandler(deviceName);
		packetHandler_ = dynamixel::PacketHandler::getPacketHandler(protocolVersion);
		group_ = lookup_.getGroupFromNames({"X5-4"}, {"X-01059", "X-01077"} );

		if(controlStrategyInt == 1)
			control_strategy = hebi::Command::ControlStrategy::DirectPWM;
		else if(controlStrategyInt == 2)
			control_strategy = hebi::Command::ControlStrategy::Strategy2;
		else{
			RSINFO_IF(visualizable_, "selected control strategy 1 (direct PWM) or 2: ")
			
		}

		//RSINFO_IF(visualizable_, "selected control strategY: " << control_strategy)

		//cmd_ = hebi::GroupCommand(2);

		if (!group_){
			RSINFO_IF(visualizable_, "No group found!");
			ActuatorConnected = false;
		}
		else{
			num_modules_ = group_->size();

			if(control_strategy == hebi::Command::ControlStrategy::Strategy2){
				for (int module_index = 0; module_index < num_modules_; module_index++){
					cmd_[module_index].settings().actuator().controlStrategy().set(control_strategy);
					//positionGain() return a reference to an object CommandGain that is an using to the class Gains templated with its proper argmunts
					//When the class has been templated, I can use its methods
					cmd_[module_index].settings().actuator().positionGains().kP().set(80);
					cmd_[module_index].settings().actuator().positionGains().kI().set(0.0);
					cmd_[module_index].settings().actuator().positionGains().kD().set(0.09);

					cmd_[module_index].settings().actuator().velocityGains().kP().set(0.0);
					cmd_[module_index].settings().actuator().velocityGains().kI().set(0.0);
					cmd_[module_index].settings().actuator().velocityGains().kD().set(0.0);

					cmd_[module_index].settings().actuator().effortGains().kP().set(0.0);
					cmd_[module_index].settings().actuator().effortGains().kI().set(0.0);
					cmd_[module_index].settings().actuator().effortGains().kD().set(0.0);

					cmd_[module_index].settings().actuator().springConstant().set(75.0);

					//operator [] access the command from the group command
				}	
			}		
			else if(control_strategy == hebi::Command::ControlStrategy::DirectPWM){
				for (int module_index = 0; module_index < num_modules_; module_index++){
					cmd_[module_index].settings().actuator().controlStrategy().set(control_strategy);
				}
			}

			group_->setCommandLifetimeMs(10); //0.01s
			group_->setFeedbackFrequencyHz(100);  //it's important to receive all the feedback at the same time
			ActuatorConnected = true; 
		
			//Enable dynamixel by rising the EnableTorque bit in the corresponding address

			//it introduces asynchrony among multi-modal
			//inputs for the RL policy: there is misalignment between the
			//proprioceptive state and the visual observation in the real
			//robot
			num_pos_ = num_pos;
			num_vel_ = num_vel;
			joint_history_pos_.setZero(num_pos*3);
			joint_history_vel_.setZero(num_vel*3);
			
			int dxl_comm_result = packetHandler_->write1ByteTxRx(portHandler_, dxl_id, addrTorqueEnable, TorqueEnable, &dxl_error_);
		}
	}

	~Actuators(){};

	void sendCommand(Eigen::VectorXd pTarget3, Eigen::Vector3d motorTorque){
		
		if(control_strategy == hebi::Command::ControlStrategy::DirectPWM)
			cmd_.setEffort(motorTorque.head(2));
		else if(control_strategy == hebi::Command::ControlStrategy::DirectPWM)
			
		group_->sendCommand(cmd_);
		//the size of the byte that you should write is visible in the control table
		int dxl_comm_result = packetHandler_->write4ByteTxRx(portHandler_, dxl_id, addrGoalPosition, pTarget3[2], &dxl_error_);

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

	Eigen::VectorXd getFeedback(){

		double dxl_pos;
		double dxl_vel;

		//Da segmentation fault se i motori non sono connessi e la chiami lo stesso
		group_->sendFeedbackRequest(); // Sends a request to the modules for feedback and immediately returns

		group_->getNextFeedback(Gfeedback_);

		const double m1_pos_fbk = Gfeedback_[0].actuator().position().get();
		const double m2_pos_fbk = Gfeedback_[1].actuator().position().get();
		packetHandler_->read4ByteTxRx(portHandler_, dxl_id, addrPresentPosition, (uint32_t*)&dxl_pos, &dxl_error_);
		Eigen::Vector3d motor_pos;
		motor_pos << m1_pos_fbk, m2_pos_fbk, dxl_pos;

		const double m1_vel_fbk = Gfeedback_[0].actuator().velocity().get();
		const double m2_vel_fbk = Gfeedback_[1].actuator().velocity().get();
		packetHandler_->read4ByteTxRx(portHandler_, dxl_id, addrPresentVelocity, (uint32_t*)&dxl_vel, &dxl_error_);
		Eigen::Vector3d motor_vel;
		motor_vel << m1_vel_fbk, m2_vel_fbk, dxl_vel;

		updateJointHistory(motor_pos, motor_vel);

		//Eigen::Matrix<double, 2, 3> gyros = Gfeedback_.getGyro();
		const auto real_orientation_ = Gfeedback_[0].imu().orientation().get();
		const auto imu_ang_vel = Gfeedback_[0].imu().gyro().get();
		const auto imu_lin_vel = Gfeedback_[1].imu().gyro().get();

		quat_[0] = real_orientation_.getW();
		quat_[1] = real_orientation_.getX();
		quat_[2] = real_orientation_.getY();
		quat_[3] = real_orientation_.getZ();
		raisim::quatToRotMat(quat_, rot_);
		Eigen::Matrix3d m = rot_.e();
		swapMatrixRows(m);

		//R0 is the base orientation
		//The linear and angular velocity are expressed with respect the current orientation frame. But if it's going ahead, it is going ahead with respect the 
		//rotated frame. So I must bring the velocity vector to the body frame
		Eigen::Vector3d imu_ang_vel_eig;
		imu_ang_vel_eig << imu_ang_vel.getX(), imu_ang_vel.getY(), imu_ang_vel.getZ();
		bodyAngularVel_ = m.transpose()*imu_ang_vel_eig;
		//bodyLinearVel_ = m.transpose()*imu_lin_vel;

		Eigen::VectorXd observations;
		//initialize always the 
		observations.setZero(9 + joint_history_pos_.size() + joint_history_vel_.size());
		observations << m.row(1).transpose(),
				m.row(2).transpose(),
				bodyAngularVel_,
				joint_history_pos_, 
				joint_history_vel_;

		return observations;
	}     

	void updateJointHistory(Eigen::Vector3d motor_pos, Eigen::Vector3d motor_vel){
		
		joint_history_pos_.head(joint_history_pos_.size() - 3) = joint_history_pos_.tail(joint_history_pos_.size() - 3);
		joint_history_vel_.head(joint_history_pos_.size() - 3) = joint_history_vel_.tail(joint_history_pos_.size() - 3);
		
		joint_history_pos_.tail(3) = motor_pos;
		joint_history_vel_.tail(3) = motor_vel;
	}

	void checkMotors(){};

 private:
	dynamixel::PortHandler *portHandler_;
	dynamixel::PacketHandler *packetHandler_;
	uint8_t dxl_error_ = 0;

	//hebi
	bool ActuatorConnected_ = true;
	int num_modules_;
	hebi::Lookup lookup_;
	std::shared_ptr<hebi::Group> group_;
	hebi::GroupCommand cmd_ = hebi::GroupCommand(2);
	hebi::GroupFeedback Gfeedback_ = hebi::GroupFeedback(2);
	hebi::Command::ControlStrategy control_strategy;  //ControlStrategy it's an enum

	raisim::Vec<4> quat_;
	raisim::Mat<3,3> rot_;

	Eigen::Vector3d bodyAngularVel_;
	//Eigen::Vector3d bodyLinearVel_;

	Eigen::VectorXd joint_history_pos_;
	Eigen::VectorXd joint_history_vel_;

	int num_pos_;
	int num_vel_;
	
};

#endif