import os, glob
import matplotlib.pyplot as plt

FOLDER_PATH = "/user/frosa/Multi-Task-LFD-Framework/repo/mimic-play/MimicPlay/trained_models_lowlevel_demo_human_agent_ur5e_3D_state_same_conf_with_valid_modes_5"


if __name__ == "__main__":
    epoch_folders = glob.glob(os.path.join(FOLDER_PATH, "rollout_*_epoch_*"))
    # order the folders by epoch number
    epoch_folders.sort(key=lambda x: int(x.split("epoch_")[-1]))
    
    run_performance = {}
    for epoch_folder in epoch_folders:
        
        log_file = os.path.join(epoch_folder, "log.txt")
        if not os.path.exists(log_file):
            continue
        
        epoch_number = int(epoch_folder.split("epoch_")[-1])
        print(f"Processing folder for epoch {epoch_number}")
        run_performance[epoch_number] = {}
        run_performance[epoch_number]["success"] = 0
        run_performance[epoch_number]["picked"] = 0
        run_performance[epoch_number]["reached"] = 0
        with open(log_file, "r") as f:
            lines = f.readlines()
            result_line = lines[-1].strip()
            print(f"Result line: {result_line}")
            reached_rate = float(result_line.split("Reached rate: ")[-1].split(",")[0].strip("%"))
            picked_rate = float(result_line.split("Picked rate: ")[-1].split(",")[0].strip("%"))
            success_rate = float(result_line.split("Success rate: ")[-1].strip("%"))
            run_performance[epoch_number]["reached"] = reached_rate
            run_performance[epoch_number]["picked"] = picked_rate
            run_performance[epoch_number]["success"] = success_rate
            print(f"Extracted rates - Reached: {reached_rate}%, Picked: {picked_rate}, Success: {success_rate}")
            
    # create a histogram of reached, picked, and success rates across epochs
    epochs = list(run_performance.keys())
    reached_rates = [run_performance[epoch]["reached"] for epoch in epochs]
    picked_rates = [run_performance[epoch]["picked"] for epoch in epochs]
    success_rates = [run_performance[epoch]["success"] for epoch in epochs]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, reached_rates, label="Reached Rate")
    plt.plot(epochs, picked_rates, label="Picked Rate")
    plt.plot(epochs, success_rates, label="Success Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Rate")
    plt.title("Performance Rates Across Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(FOLDER_PATH, "performance_rates_across_epochs.png"))
    plt.show()
    