#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <sstream>

using namespace std::chrono_literals;


enum class RobotType {
    ANYMAL,
    IIWA,
    ARMANDO
};


std::vector<std::string> getJointNames(RobotType robotype){

	if(robotype == RobotType::ANYMAL)
		return  {"LF_HAA", "LF_HFE", "LF_KFE", "RF_HAA", "RF_HFE", "RF_KFE", "LH_HAA", "LH_HFE", "LH_KFE", "RH_HAA", "RH_HFE", "RH_KFE"};
	else if(robotype == RobotType::IIWA)
		return {"joint_a1", "joint_a2", "joint_a3", "joint_a4", "joint_a5", "joint_a6", "joint_a7"};
	else if(robotype == RobotType::ARMANDO)
		return {"j0", "j1", "j2", "j3"};

}


void setPoints(trajectory_msgs::msg::JointTrajectoryPoint *point, const std::vector<float> &joints_value, const std::vector<std::string> &jointNames){
	
	std::vector<std::string> jointNames_string(std::begin(jointNames), std::end(jointNames));
	point->positions.resize(jointNames_string.size());

	int i = 0;
	for(const auto & j:joints_value){
		point->positions[i] = j;
		i++;
	}

}


int main(int argc, char * argv[])
{
	
	rclcpp::init(argc, argv);

	std::vector<float> joints_value;
	for(int i = 0; i<argc-2; i++){
		try {
			joints_value.push_back(std::stof(argv[i+2]));
		}catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << argv[i + 2] << " is not a valid float." << std::endl;
            return 1;
		}
	}

    RobotType robotType = RobotType::ARMANDO;

	std::string argv1 = argv[1]; //to prevent warnings
	if(argv1=="iiwa")
		robotType = RobotType::IIWA;
	else if(argv1=="anymal")
		robotType = RobotType::ANYMAL;
	else if(argv1=="armando" || std::string(argv[1]).empty())
		robotType = RobotType::ARMANDO;
	

	std::shared_ptr<rclcpp::Node> node = std::make_shared<rclcpp::Node>("arm_test");

	auto publisher = node->create_publisher<trajectory_msgs::msg::JointTrajectory>(
		"/joint_trajectory_controller/joint_trajectory", 10);

	RCLCPP_INFO(node->get_logger(), "node created");


	trajectory_msgs::msg::JointTrajectory command;
	trajectory_msgs::msg::JointTrajectoryPoint point;

	std::vector<std::string> jointNames = getJointNames(robotType);

	command.joint_names = jointNames;

	command.points.resize(1); //we dont have a set of set points, but just once.
	command.points[0].positions.resize(jointNames.size());

	setPoints(&point, joints_value, jointNames);

	/*command.points[0].positions[0] = 0.2;
	command.points[0].positions[1] = 0.2;
	command.points[0].positions[2] = 0.2;
	command.points[0].positions[3] = 0.2;*/
	command.points[0] = point;

	command.points[0].time_from_start = rclcpp::Duration::from_seconds(1.0);  // start asap

	std::ostringstream oss;
	for (const auto& val : joints_value) {
		oss << val << ", ";
	}


	while (1) {
			publisher->publish(command);
			RCLCPP_INFO(node->get_logger(), "sending: [%s]", oss.str().c_str());
			std::this_thread::sleep_for(100ms);
			rclcpp::spin_some(node);
	}

	//attento che dentro spin_some ci va l'istanza della classe, in spin() ci può andare il std::make_shared<class>()
	//La differenza è che in spin_some cerca se può lanciare una callback, e poi va avanti con il codice
	//per questo sta nei while.
	//Spin invece è BLOCCANTE quindi cerca continuamente se arrivano callback
	//per questo motivo puoi fare:
	//rclcpp::spin(std::make_shared<class>()) -> perché qui chiama il costruttore e si BLOCCA in attesa di callback
	//rcpcpp:spin_some() non si blocca. Se ci mettessi di nuovo std::make_shared<class>() chiamerebbe ogni volta il costruttore
	//per farlo funzionare, e bloccare sulle callback della classe, devo metterci l'istanza della classe, quindi dichiaro prima il nodo (istanziandolo)
	//e poi lo passo


	rclcpp::shutdown();

	return 0;
}
/*ros2 topic pub -r 10 /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
	header: {
		stamp: { sec: 0, nanosec: 0 },
		frame_id: ''
	},
	joint_names: ['LF_HAA', 'LF_KFE'],
	points: [{
		positions: [0.0, 0.2],
		velocities: [],
		time_from_start: { sec: 1, nanosec: 0 }
	}]
}"*/