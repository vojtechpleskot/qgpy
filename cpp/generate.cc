#include <iostream>
#include "Pythia8/Pythia.h"
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include "cxxopts.hpp" // Assuming you have this file

#include "fastjet/ClusterSequence.hh"
#include "fastjet/PseudoJet.hh"
// #include "IFNPlugin.hh"
#include "fastjet/contrib/IFNPlugin.hh"
// #include "FlavNeutraliser.hh"
#include "fastjet/contrib/FlavNeutraliser.hh"

#include "HepMC3/GenEvent.h"
#include "HepMC3/WriterAscii.h"
#include "HepMC3/FourVector.h"
#include "HepMC3/GenParticle.h"
#include "Pythia8Plugins/HepMC3.h"
#include "fastjet/ClusterSequenceArea.hh"
#include "Pythia8Plugins/FastJet3.h"

using namespace Pythia8;
using namespace std;
using namespace fastjet;
using namespace fastjet::contrib;

// Helper function to remove square brackets from flavour description
string strip_brackets(const string& s) {
    if (s.size() >= 2 && s.front() == '[' && s.back() == ']') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

// Helper function to determine the flavour of a particle from the FlavInfo object.
int get_flavour(const FlavInfo& flav_info) {
    if (flav_info[6] != 0) return 6;
    if (flav_info[5] != 0) return 5;
    if (flav_info[4] != 0) return 4;
    if (flav_info[3] != 0) return 3;
    if (flav_info[2] != 0) return 2;
    if (flav_info[1] != 0) return 1;
    if (flav_info[0] != 0) return 21; // gluon
    return -1;
}

// Helper quark identifier.
bool is_quark(int flavour) {
    return flavour >= 1 && flavour <= 6;
}

int main(int argc, char* argv[]) {

    // Command line options parsing
    cxxopts::Options options("60jets", "Pythia8 and FastJet example with IFNPlugin");

    options.add_options()
        ("c,compare", "Create a second jet collection and compare it with the default one", cxxopts::value<bool>()->default_value("false"))
        ("n,nevents", "Number of events to generate", cxxopts::value<int>()->default_value("10"))
        ("s,seed", "Random seed to initialize Pythia with", cxxopts::value<int>()->default_value("0"))
        ("phmin,pTHatMin", "Minimum pT hat for hard processes", cxxopts::value<double>()->default_value("-1."))
        ("phmax,pTHatMax", "Maximum pT hat for hard processes", cxxopts::value<double>()->default_value("-1."))
        ("o,output", "Output file name", cxxopts::value<std::string>()->default_value("labels"))
        ("r,recoJetPtMin", "Minimum pT for jets", cxxopts::value<double>()->default_value("5.0"))
        ("h,help", "Print usage")
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }
    bool compare = result["compare"].as<bool>();
    int nevents = result["nevents"].as<int>();
    int seed = result["seed"].as<int>();
    double pTHatMin = result["pTHatMin"].as<double>();
    double pTHatMax = result["pTHatMax"].as<double>();
    std::string output = result["output"].as<std::string>();
    double recoJetPtMin = result["recoJetPtMin"].as<double>();

    // Print the parsed options
    std::cout << "Parsed options:" << std::endl;
    std::cout << "  Compare: " << (compare ? "true" : "false") << std::endl;
    std::cout << "  Number of events: " << nevents << std::endl;
    std::cout << "  Seed: " << seed << std::endl;
    std::cout << "  pTHatMin: " << pTHatMin << std::endl;
    std::cout << "  pTHatMax: " << pTHatMax << std::endl;
    std::cout << "  Output file: " << output << std::endl;
    std::cout << "  Jet pT min: " << recoJetPtMin << std::endl;

    // Create Pythia instance
    Pythia pythia;

    // Set the seed of the random number generator
    // From the Pythia manual: "A negative value gives the default seed,
    // a value 0 gives a random seed based on the time, and
    // a value between 1 and 900,000,000 a unique different random number sequence."
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + std::to_string(seed));

    // Proton-proton collisions
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = 13600");

    // Enable hard QCD processes
    pythia.readString("HardQCD:all = on");
    if (pTHatMin > 0) {
        pythia.readString("PhaseSpace:pTHatMin = " + std::to_string(pTHatMin));
    }
    if (pTHatMax > 0) {
        pythia.readString("PhaseSpace:pTHatMax = " + std::to_string(pTHatMax));
    }

    // Turn ON hadronization and parton-level evolution
    pythia.readString("PartonLevel:all = on");
    pythia.readString("HadronLevel:all = on");
    
    // Suppress Pythia's event record printout
    pythia.readString("Next:numberShowEvent = 0");

    // Initialize Pythia
    pythia.init();

    // Create a HepMC event writer
    HepMC3::Pythia8ToHepMC3 toHepMC;
    HepMC3::WriterAscii writer(output + ".hepmc3");

    ofstream outfile(output + ".txt");
    outfile << "event_number\tpt\t\tE\t\trap\tphi\tflav\tATLAS_tag\n";

    // Jet definition with flavour recombiner
    JetDefinition base_jet_def(antikt_algorithm, 0.4);
    fastjet::contrib::FlavRecombiner flav_recombiner;
    base_jet_def.set_recombiner(&flav_recombiner);

    // Set IFN plugin parameters
    double alpha = 2.0;
    double omega = 3.0 - alpha;
    fastjet::contrib::FlavRecombiner::FlavSummation flav_summation = fastjet::contrib::FlavRecombiner::net;

    auto ifn_plugin = new fastjet::contrib::IFNPlugin(base_jet_def, alpha, omega, flav_summation);
    JetDefinition IFN_jet_def(ifn_plugin);
    IFN_jet_def.delete_plugin_when_unused();

    int counter = 0;
    int n_jets = 0;

    for (int i = 0; i < nevents; i++) {
        if (!pythia.next()) continue;

        // Write the event to HepMC
        HepMC3::GenEvent genEvent;
        toHepMC.fill_next_event(pythia.event, &genEvent);
        writer.write_event(genEvent);        

        // Select partons with status 61–69
        vector<PseudoJet> selected_partons;

        for (int j = 0; j < pythia.event.size(); j++) {
            const Particle& parton = pythia.event[j];
            int status = abs(parton.status());
            int id = abs(parton.id());

            // Select partons with status 61–69 and id = 1–6 or 21
            if (status >= 61 && status <= 69 &&
                ((id >= 1 && id <= 6) || id == 21)) {

                double px = parton.px();
                double py = parton.py();
                double pz = parton.pz();
                double E  = parton.e();
                PseudoJet p(px, py, pz, E);
                p.set_user_info(new FlavHistory(parton.id()));
                selected_partons.push_back(p);
            }
        }

        // Run IFN clustering
        vector<PseudoJet> IFN_jets = IFN_jet_def(selected_partons);

        for (unsigned int ijet = 0; ijet < IFN_jets.size(); ijet++) {
            const auto& jet = IFN_jets[ijet];
            if (jet.pt() < recoJetPtMin) continue;

            int ifn_label = get_flavour(FlavHistory::current_flavour_of(jet));

            auto constituents = sorted_by_E(jet.constituents());
            int atlas_label = -1;
            if (!constituents.empty()) {
                atlas_label = get_flavour(FlavHistory::initial_flavour_of(constituents.front()));
                ++ n_jets;
                if ((is_quark(atlas_label) && !is_quark(ifn_label))){
                    ++counter;
                }
            }

            outfile << i + 1 << "\t"
                    << fixed << setprecision(3)
                    << jet.pt()  << "\t"
                    << jet.e()   << "\t"
                    << jet.rap() << "\t"
                    << jet.phi() << "\t"
                    // << "g: " << FlavHistory::current_flavour_of(jet)[0] << "\t"
                    // << "d: " << FlavHistory::current_flavour_of(jet)[1] << "\t"
                    // << "u: " << FlavHistory::current_flavour_of(jet)[2] << "\t"
                    // << "s: " << FlavHistory::current_flavour_of(jet)[3] << "\t"
                    // << "c: " << FlavHistory::current_flavour_of(jet)[4] << "\t"
                    // << "b: " << FlavHistory::current_flavour_of(jet)[5] << "\t"
                    // << "t: " << FlavHistory::current_flavour_of(jet)[6] << "\t"
                    << ifn_label << "\t"
                    << atlas_label << "\n";
        }
    }

    std::cout << "Processed " << nevents << " events." << std::endl;
    std::cout << "Found " << counter << " jets with the atlas label different from the IFN label, from " << n_jets << " checked jets." << std::endl;

    outfile.close();
    writer.close();
    pythia.stat();
    return 0;
}

