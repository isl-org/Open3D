// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Config.h"
#include "LegacyReconstructionUtil.h"
#include "open3d/Open3D.h"

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > LegacyReconstruction [options]");
    utility::LogInfo("      Given an RGBD image sequence, perform the following steps:");
    utility::LogInfo("      1. Make fragments from the RGBD image sequence.");
    utility::LogInfo("      2. Register multiple fragments.");
    utility::LogInfo("      3. Refine rough registration.");
    utility::LogInfo("      4. Integrate the whole RGBD sequence to make final mesh or point clouds.");
    utility::LogInfo("      5. (Optional) Run slac optimization for fragments.");
    utility::LogInfo("      6. (Optional) Run slac integration for sequence.");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --config [path to config json file]");
    utility::LogInfo("    --default_dataset [(optional) default dataset to be used, only if the config file is not provided]");
    utility::LogInfo("                      [options: (lounge, bedroom, jack_jack), default: lounge]");
    utility::LogInfo("    --make");
    utility::LogInfo("    --register");
    utility::LogInfo("    --refine");
    utility::LogInfo("    --integrate");
    utility::LogInfo("    --slac");
    utility::LogInfo("    --slac_integrate");
    utility::LogInfo("    --debug_mode [turn on debug mode]");
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    using namespace open3d;
    using namespace open3d::apps::offline_reconstruction;

    if (argc < 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    // Load dataset.
    Json::Value config;
    if (utility::ProgramOptionExists(argc, argv, "--config")) {
        bool ret;
        try {
            const std::string config_path = utility::GetProgramOptionAsString(
                    argc, argv, "--config", "");
            utility::LogInfo("Loading config file: {}", config_path);
            ret = ReadJsonFromFile(config_path, config);

        } catch (const std::exception& e) {
            utility::LogWarning("Failed to load config file: {}", e.what());
            return 1;
        }
        InitConfig(config);
        ret = CheckFolderStructure(config["path_dataset"].asString());
        if (!ret) {
            utility::LogWarning("Check folder structure failed.");
            return 1;
        }
    } else {
        config = DefaultDatasetLoader(utility::GetProgramOptionAsString(
                argc, argv, "--default_dataset", "lounge"));
    }

    if (utility::ProgramOptionExists(argc, argv, "--debug_mode")) {
        config["debug_mode"] = true;
    } else {
        config["debug_mode"] = false;
    }

    config["device"] =
            utility::GetProgramOptionAsString(argc, argv, "--device", "CPU:0");

    // Print configuration in console.
    utility::LogInfo("====================================");
    utility::LogInfo("Configuration");
    utility::LogInfo("====================================");
    for (Json::Value::const_iterator it = config.begin(); it != config.end();
         ++it) {
        utility::LogInfo("{}: {}", it.key().asString(), it->asString());
    }
    utility::LogInfo("====================================");

    // Init Reconstruction pipeline.
    ReconstructionPipeline pipeline(config);

    utility::Timer timer;
    std::array<double, 6> durations({0, 0, 0, 0, 0, 0});
    if (utility::ProgramOptionExists(argc, argv, "--make")) {
        timer.Start();
        pipeline.MakeFragments();
        timer.Stop();
        const double ms = timer.GetDurationInMillisecond();
        durations[0] = ms;
    }
    if (utility::ProgramOptionExists(argc, argv, "--register")) {
        timer.Start();
        pipeline.RegisterFragments();
        timer.Stop();
        const double ms = timer.GetDurationInMillisecond();
        durations[1] = ms;
    }
    if (utility::ProgramOptionExists(argc, argv, "--refine")) {
        timer.Start();
        pipeline.RefineRegistration();
        timer.Stop();
        const double ms = timer.GetDurationInMillisecond();
        durations[2] = ms;
    }
    if (utility::ProgramOptionExists(argc, argv, "--integrate")) {
        timer.Start();
        pipeline.IntegrateScene();
        timer.Stop();
        const double ms = timer.GetDurationInMillisecond();
        durations[3] = ms;
    }
    if (utility::ProgramOptionExists(argc, argv, "--slac")) {
        timer.Start();
        pipeline.SLAC();
        timer.Stop();
        const double ms = timer.GetDurationInMillisecond();
        durations[4] = ms;
    }
    if (utility::ProgramOptionExists(argc, argv, "--slac_integrate")) {
        timer.Start();
        pipeline.IntegrateSceneSLAC();
        timer.Stop();
        const double ms = timer.GetDurationInMillisecond();
        durations[5] = ms;
    }

    utility::LogInfo("====================================");
    utility::LogInfo("Elapsed time (in h:m:s)");
    utility::LogInfo("====================================");
    utility::LogInfo("Making fragments:      {}", DurationToHMS(durations[0]));
    utility::LogInfo("Register fragments:    {}", DurationToHMS(durations[1]));
    utility::LogInfo("Refining registration: {}", DurationToHMS(durations[2]));
    utility::LogInfo("Integrating frames:    {}", DurationToHMS(durations[3]));
    utility::LogInfo("SLAC:                  {}", DurationToHMS(durations[4]));
    utility::LogInfo("SLAC integration:      {}", DurationToHMS(durations[5]));

    return 0;
}
