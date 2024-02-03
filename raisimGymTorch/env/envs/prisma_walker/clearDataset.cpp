#include <iostream>
#include <fstream>
#include <string>

using namespace std;


int main(){

    ifstream m1, m2;
    ofstream m1_new, m2_new;
    float current_pos;
    float old_pos = 0;
    m1.open("pos_m1_18s.txt", std::ios::in);
    m1_new.open("m1.txt", std::ios::out);

    m2.open("pos_m2_18s.txt", std::ios::in);
    m2_new.open("m2.txt", std::ios::out);

    while (m1>> current_pos) { ///each time is called read the next value
        
        if((current_pos - old_pos)>1e-4){
            m1_new << current_pos <<"\n";
        }

        old_pos = current_pos;

    }

    old_pos = 0;
    while (m2>> current_pos) { ///each time the ifstream object is called, it reads the next value
        
        if((current_pos - old_pos)>1e-4){
            m2_new << current_pos <<"\n";
        }

        old_pos = current_pos;

    }


    return 0;


}