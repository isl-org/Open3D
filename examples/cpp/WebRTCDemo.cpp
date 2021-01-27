// #include "open3d/visualization/webrtc/WebRTCStreamer.h"

// int main() {
//     WebRTCStreamer wss;
//     wss.Run();
// }

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

class Dog {
public:
    Dog(const std::string& name) : name_(name) {}
    void PrintName() const { std::cout << "Dog: " << name_ << std::endl; }
    std::string name_;
};

void Rename(Dog& dog) {
    std::string new_name = "feifei";
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (dog.name_ == "feifei") {
            dog.name_ = "fafa";
        } else {
            dog.name_ = "feifei";
        }
    }
}

void Print(Dog& dog) {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        dog.PrintName();
    }
}

int main() {
    Dog dog("feifei");
    std::thread first(Rename, std::ref(dog));
    std::thread second(Print, std::ref(dog));

    std::cout << "main, rename and print now execute concurrently...\n";
    first.join();   // pauses until first finishes
    second.join();  // pauses until second finishes
    std::cout << "rename and print completed.\n";

    return 0;
}
