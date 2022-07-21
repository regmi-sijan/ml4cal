#include <iostream>
#include "TFile.h"

using namespace std;

int main(int argc, char* argv[]) {

    TFile f("demo.root");

    if (f.IsZombie()) {
        cout << "Error opening file" << endl;
        exit(-1);
    }
}
