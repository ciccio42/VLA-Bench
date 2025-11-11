import os
import json
import argparse
import glob
import numpy as np

BLACK_LIST = [] #[0,5,10,15]
OBJ_3_COLOR_LIST = ["green", "orange", "red", "grey"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze results from robosuite experiments.")
    parser.add_argument("--path", type=str, required=True, help="Path to the directory containing result files.")
    parser.add_argument("--task_name", type=str, default="pick_place", help="Name of the task to analyze.")
    parser.add_argument("--change_obj_pos", type=str, default="False", help="Whether to change object positions.")
    parser.add_argument("--ood", type=str, default="False", help="Whether to change object positions.")
    parser.add_argument("--change_command", type=str, default="False", help="Whether to change object positions.")
    parser.add_argument("--obj_set", type=str, default="-1", help="Object set identifier.")
    
    args = parser.parse_args()
    print(f"rollout_{args.task_name}_*_{args.change_obj_pos}_obj_set_{args.obj_set}_change_command_{args.change_command}_*_OoD_{args.ood}")
    result_folders = glob.glob(os.path.join(args.path, f"rollout_{args.task_name}_*_{args.change_obj_pos}_obj_set_{args.obj_set}_change_command_{args.change_command}_OoD_{args.ood}"))

    if not result_folders:
        print("No result folders found in the specified path.")
        exit(1)

    result = dict()
    for folder in result_folders:
        run_number = folder.split(f'rollout_{args.task_name}_')[-1].split('_')[0]
        print(f"Processing folder: {folder} (Run {run_number})")
        result[run_number] = dict()
        
        json_files = glob.glob(os.path.join(folder, "*.json"))
        json_files.sort(key= lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
        for i, file_path in enumerate(json_files):
            # print(f"{file_path.split('/')[-1]}")
            with open(file_path, 'r') as file:
                data = json.load(file)
                variation_id = int(data.get("variation_id", 0))
                
                if args.obj_set != "-1":
                    color = OBJ_3_COLOR_LIST[int(variation_id/len(OBJ_3_COLOR_LIST))]
                    # print(f"Color {color} - Variation id {variation_id}")
                    
                if variation_id in BLACK_LIST:
                    print(f"Skipping variation_id {variation_id}")
                    continue

                # print(f"Results from {file_path}:")
                # print(json.dumps(data, indent=4))
                if len(result[run_number]) == 0:
                    for metric in data.keys():
                        result[run_number][metric] = []
                        if args.obj_set != "-1":
                            for color in OBJ_3_COLOR_LIST:
                                if color not in result[run_number].keys():
                                    result[run_number][color] = dict()
                                if metric not in result[run_number][color].keys():
                                    result[run_number][color][metric] = []
                          
                for metric in data.keys():                  
                    # print(f"{color}-{metric}")
                    value = data[metric]
                    result[run_number][metric].append(value)
                    if args.obj_set != "-1":
                        result[run_number][color][metric].append(value)

    # Average the results for each run
    aggregated_results = dict()
    for run in result.keys():
        # print(f"Run: {run}")
        for metric_color in result[run].keys():
            # print(f"\tMetric-color: {metric_color}")
            if isinstance(result[run][metric_color], list):
                # print(f"\t\tMetric: {metric_color}")    
                # metric_color is metric
                values = result[run][metric_color]
                if metric_color not in aggregated_results:
                    aggregated_results[metric_color] = []
                aggregated_results[metric_color].append(np.mean(values))    
            elif isinstance(result[run][metric_color], dict):
                # metric_color is color
                # print(f"\t\tColor: {metric_color}")    
                if metric_color not in aggregated_results:
                    aggregated_results[metric_color] = dict()
                    
                for metric in result[run][metric_color].keys():
                    values = result[run][metric_color][metric]
                    if metric not in aggregated_results[metric_color].keys():
                        aggregated_results[metric_color][metric] = []
                    aggregated_results[metric_color][metric].append(np.mean(values))
    print(aggregated_results)

    final_results = dict()
    
    for metric_color in aggregated_results.keys():
        if metric_color not in final_results.keys():
            if isinstance(aggregated_results[metric_color], list):
                final_results[metric_color] = (round(np.mean(aggregated_results[metric_color]), 3), round(np.std(aggregated_results[metric_color]), 3))
            elif isinstance(aggregated_results[metric_color], dict):
                final_results[metric_color] = dict()
                for metric in aggregated_results[metric_color].keys():
                    final_results[metric_color][metric] = (round(np.mean(aggregated_results[metric_color][metric]), 3), round(np.std(aggregated_results[metric_color][metric]), 3))
        
    # Average across runs
    # print(f"Success values {aggregated_results['success']} ")
    # final_results = {metric: (round(np.mean(values), 3), round(np.std(values), 3)) for metric, values in aggregated_results.items()}

    # print("Final Results:")
    # for metric, (mean, std) in final_results.items():
    #     print(f"{metric}: Mean = {mean:.4f}, Std = {std:.4f}")
    # Save final results to a JSON file
    output_file = os.path.join(args.path, f"final_results_{args.task_name}_{args.change_obj_pos}_OoD_{args.ood}_obj_set_{args.obj_set}.json")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=4)    
            