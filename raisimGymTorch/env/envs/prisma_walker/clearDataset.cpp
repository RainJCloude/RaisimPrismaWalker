#include <iostream>
#include <fstream>
#include <string>
#include <thread>
 
using namespace std;


int main(){

    ifstream m1, m2;
    ofstream m1_new, m2_new;
    float current_pos;
    float old_pos = 0;
    m1.open("pos_m1_18s.txt", std::ios::in);
    m2.open("pos_m2_18s.txt", std::ios::in);

    int length_m1 = 0;
    int length_m2 = 0;

    double threshold = 0.0001;

    do{
        m1_new.open("m1.txt", std::ios::out);
        old_pos = 0;
        if(length_m1 > 1000) //too much, rise the threshold
            threshold = threshold + 0.000001;
        else
            threshold = threshold - 0.000001;        
        length_m1 = 0;
        while (m1>> current_pos) { ///each time is called read the next value
            if((current_pos - old_pos)>threshold){
                m1_new << current_pos <<"\n";
                length_m1++;
            }
            old_pos = current_pos;
        }
        m1_new.close();
        m1.clear(); //to restart with the file from the start
        m1.seekg(0);
        std::cout<<"campioni m1: "<<length_m1<<endl;
    }while(length_m1 < 1000);
    
    threshold = 0.0001;
    do{
        m2_new.open("m2.txt", std::ios::out);
        old_pos = 0;
        if(length_m2 > 1000) //too much, rise the threshold
            threshold = threshold + 0.000001;
        else
            threshold = threshold - 0.000001;
        length_m2 = 0;
        while (m2>> current_pos) { ///each time is called read the next value
            if((current_pos - old_pos)>threshold){
                m2_new << current_pos <<"\n";
                length_m2++;
            }
            old_pos = current_pos;
        }
        m2_new.close();
        m2.clear(); //to restart with the file from the start
        m2.seekg(0);
        std::cout<<"campioni m2: "<<length_m1<<endl;
    }while(length_m2 != length_m1 );
    

    return 0;


}