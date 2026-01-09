#!/bin/bash

mkdir -p "$1"
zip -r "$1/responses.zip" responses_tmp
zip -r "$1/prompts.zip" prompts_tmp
zip -r "$1/reward_analysis.zip" reward_analysis_txt
cp "stats.json" "$1/stats.json"
zip -r "$1/dynamic_reward_functions_final.zip" "dynamic_reward_functions_final"
zip -r "$1/dynamic_reward_functions_initial.zip" "dynamic_reward_functions_initial"

# rm -rf responses_tmp
# rm -rf prompts_tmp
# rm -rf reward_analysis_txt
# rm -rf dynamic_reward_functions_final
# rm -rf dynamic_reward_functions_initial


# How to run:
# sudo bash backup_outputs.sh "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/dynamic_constraint_rewards/ceiling_lamps_above_bed_corners"