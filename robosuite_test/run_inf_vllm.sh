#!/bin/bash



# curl http://gnode02:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "nvidia/Cosmos-Reason2-2B",
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#           { "type": "videos", 
#             "videos": { "path": "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/VLA-Benchmark/robosuite_test/temp_padded_video_task_00_traj007_camera_front_image.mp4" }},
#           { "type": "text", "text": "Describe the task being performed." }
#         ]
#       }
#     ]
#   }'