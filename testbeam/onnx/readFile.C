void readFile(TString filename) {
    TFile f(filename);

    if (f.IsZombie()) {
        cout << "Error opening file" << endl;
        exit(-1);
    }

    TTree *tree = (TTree*)f.Get("trainingtree;1");
    TBranch *branch  = tree->GetBranch("waveform");

    Int_t waveform[64][32];

    branch->SetAddress(&waveform);

    Long64_t n = branch->GetEntries();

  

    for (int i=0; i<10; i++) {
        Int_t m = branch->GetEntry(i);
        for(int bin=0; bin<31; bin++) {
            cout<< waveform[27][bin] << " ";
        }
        cout  << endl;
    }
   

    //free(waveform);
//    cout << "List of Keys" << endl;
//    f.GetListOfKeys()->Print();
//    cout << "ls" << endl;
//    f.ls();

//    TTree *myTree;
//    gDirectory->GetObject("trainingtree",myTree);

//    TBranch myBranch = myTree->Branch("waveform");

    f.Close();

    cout << n << endl;
    return;
}
