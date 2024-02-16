/**
 * This file demonstrates the ability to command a group.
 */
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <vector>
/*#include "lookup.hpp"
#include "group_command.hpp"
#include "command.hpp"*/

//g++ -o output_file source_file.cpp -I/include/path -L/library/path -llibrary_name
//g++ -o torque raisimGymTorch/env/envs/prisma_walker/generateData.cpp -I//home/claudio/raisim_ws/raisimlib/hebi-cpp/src -L//home/claudio/raisim_ws/raisimlib/hebi-cpp/build -lhebic++ -lhebi -I/home/claudio/raisim_ws/raisimlib/hebi-cpp/hebi/include
//-I//home/claudio/raisim_ws/raisimlib/


int main(int argc, char* argv[]){

    std::ifstream readTorques;
    readTorques.open("torques.txt");

    std::ofstream hebiData;
    hebiData.open("hebiData.txt");

    float torque_hebi1, torque_hebi2, torque_dynamixel;
    std::vector<std::vector<float>> vv; 
        
    if(readTorques.is_open()){
        while(readTorques >> torque_hebi1 >> torque_hebi2 >> torque_dynamixel ){
            std::vector<float> torques = {torque_hebi1, torque_hebi2, torque_dynamixel};
            vv.push_back(torques);
        }
    }
    else
        return -1;

    hebi::Lookup lookup;
    std::shared_ptr<hebi::Group> group;
    group = lookup.getGroupFromNames({"X5-4"}, {"X-01059", "X-01077"} );

    if (!group){
        std::cout << "No group found!" << std::endl;
        return -1;
    }
    
    int num_modules = group->size();

     // This calls a lambda function:
    group->addFeedbackHandler(
    [](const hebi::GroupFeedback& feedback)->void //return void
      { 
        // Print out position of first module:
        const auto& position = feedback[0].actuator().position();
        const auto& velocity = feedback[0].actuator().velocity();
        if (position.has())
          std::cout << position.get() << std::endl;
        else
          std::cout << "no position feedback!" << std::endl;
        
        hebiData << position.get() << "\t" << velocity.get()<< "\n"; 

      });

    // Start 200Hz feedback loop.
    std::cout << "Starting asynchronous feedback callbacks" << std::endl;
    group->setFeedbackFrequencyHz(100);
    // Create a command object; this can be sent to the group
    hebi::GroupCommand command(num_modules);
    for(const auto & data: vv){
        command[0].actuator().effort().set(data[0]);
        command[0].actuator().effort().set(data[1]);
        group->sendCommand(command);

        //without lambda fun
        //group->getNextFeedback(feedback, 10);
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); //control_dt_
    }
        //i will send the torque, and the motor will attempt to follow that command


  // (Because these occur in a separate thread, use standard multi-threaded
  // programming practice to control access to the variables whilst avoiding
  // deadlocks)
  



  // Wait 10 seconds. This should result in about 2,000 callbacks (if you have
  // all handlers enabled above, this will result in about 4,000)
  int wait_period_s = 10;
  // Stop the feedback loop, and unrelease our callback:
  group->setFeedbackFrequencyHz(0);
  group->clearFeedbackHandlers();
    hebiData.close();

 

    // NOTE: destructors automatically clean up group command and group*/
    return 0;
    }