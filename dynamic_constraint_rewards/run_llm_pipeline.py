import hydra
import os
from omegaconf import DictConfig, OmegaConf
from steerable_scene_generation.utils.omegaconf import register_resolvers
# from universal_constraint_rewards.commons import get_all_universal_reward_functions
from dynamic_constraint_rewards.commons import verify_tests_for_reward_function, get_stats_from_initial_rewards, save_reward_functions
from dynamic_constraint_rewards.create_prompt import create_prompt_1, create_prompt_2, create_prompt_3, create_prompt_4
import json

def call_gemini(llm_instruction, llm_user_prompt):
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
    load_dotenv()

    client = genai.Client()
    llm_response_1 = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            system_instruction=llm_instruction),
        contents=llm_user_prompt,
    )
    return llm_response_1.text

def call_openai(llm_instruction, llm_user_prompt):
    from openai import AzureOpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") 
    llm_response_1 = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": llm_instruction},
            {"role": "user", "content": llm_user_prompt},
        ],
    )
    return llm_response_1.choices[0].message.content
  
def call_llm(use_openai, use_gemini, llm_instruction, llm_user_prompt):
    if use_openai:
        return call_openai(llm_instruction, llm_user_prompt)
    elif use_gemini:
        return call_gemini(llm_instruction, llm_user_prompt)
    else:
        raise ValueError("Invalid LLM provider")

def run_llm_pipeline(room_type, cfg, use_gemini=False, use_openai=False, get_stats=False, load=None):
    base_dir = os.path.dirname(__file__)
    user_prompt = cfg.algorithm.ddpo.dynamic_constraint_rewards.user_query
    llm_instruction_1, llm_user_prompt_1 = create_prompt_1(user_prompt, room_type)
    user_query = user_prompt.replace(" ", "_")
    user_query = user_query.replace(".", "")
    prompts_tmp_dir = os.path.join(base_dir, f"{user_query}_prompts_tmp")
    responses_tmp_dir = os.path.join(base_dir, f"{user_query}_responses_tmp")
    os.makedirs(prompts_tmp_dir, exist_ok=True)
    os.makedirs(responses_tmp_dir, exist_ok=True)
    with open(os.path.join(prompts_tmp_dir, "llm_instruction_1.md"), "w") as f:
        f.write(llm_instruction_1)
    with open(os.path.join(prompts_tmp_dir, "llm_user_prompt_1.md"), "w") as f:
        f.write(llm_user_prompt_1)
      
    if os.path.exists(os.path.join(responses_tmp_dir, "llm_response_1.json")):
        with open(os.path.join(responses_tmp_dir, "llm_response_1.json"), "r") as f:
            constraints = json.load(f)            
    else:        
      if use_gemini or use_openai:
        llm_response_1 = call_llm(use_gemini, use_openai, llm_instruction_1, llm_user_prompt_1)
        constraints = json.loads(llm_response_1.split("```json")[1].split("```")[0].strip())
      else:
        llm_response_1 = None
        constraints = None
      if constraints is not None:
          with open(os.path.join(responses_tmp_dir, "llm_response_1.json"), "w") as f:
              json.dump(constraints, f)
          print(f"[Ashok] Saved constraints to {os.path.join(responses_tmp_dir, 'llm_response_1.json')}")

    llm_instruction_2, llm_user_prompt_2 = create_prompt_2(user_prompt, constraints, room_type)
    with open(os.path.join(prompts_tmp_dir, "llm_instruction_2.md"), "w") as f:
        f.write(llm_instruction_2)
    with open(os.path.join(prompts_tmp_dir, "llm_user_prompt_2.md"), "w") as f:
        f.write(llm_user_prompt_2)
    
    # CALL THE LLM HERE
    if os.path.exists(os.path.join(responses_tmp_dir, "llm_response_2.json")):
        with open(os.path.join(responses_tmp_dir, "llm_response_2.json"), "r") as f:
            reward_functions = json.load(f)
    else:
        if use_gemini or use_openai:
          llm_response_2 = call_llm(use_gemini, use_openai, llm_instruction_2, llm_user_prompt_2)
          reward_functions = json.loads(llm_response_2.split("```json")[1].split("```")[0].strip())
        else:
          llm_response_2 = None
          reward_functions = None
        if reward_functions is not None:
            with open(os.path.join(responses_tmp_dir, "llm_response_2.json"), "w") as f:
                json.dump(reward_functions, f)
                
        print(f"[Ashok] Saved reward functions to {os.path.join(responses_tmp_dir, 'llm_response_2.json')}")


        
    saved_reward_functions = save_reward_functions(reward_functions, reward_code_dir=f"{user_query}_dynamic_reward_functions_initial")
    
    if not saved_reward_functions:
        raise ValueError("Failed to save reward functions")
    verified_tests = verify_tests_for_reward_function(room_type, user_query, reward_code_dir=f"{user_query}_dynamic_reward_functions_initial")
    if not verified_tests:
        raise ValueError("Failed to verify tests for reward functions")
    
    if get_stats:
        dataset_stats, baseline_stats = get_stats_from_initial_rewards(reward_functions, cfg, load=load, reward_code_dir=f"{user_query}_dynamic_reward_functions_initial")
        if dataset_stats is None or baseline_stats is None:
            raise ValueError("Failed to get stats from initial rewards")
        
        print(f"Dataset stats: {dataset_stats}")
        print(f"Baseline stats: {baseline_stats}")
    
    # Stats shoulde be saved to txt files by now
    llm_instruction_3, llm_user_prompt_3 = create_prompt_3(user_prompt, constraints, reward_functions, room_type)
    
    with open(os.path.join(prompts_tmp_dir, "llm_instruction_3.md"), "w") as f:
        f.write(llm_instruction_3)
    with open(os.path.join(prompts_tmp_dir, "llm_user_prompt_3.md"), "w") as f:
        f.write(llm_user_prompt_3)
    
    # CALL THE LLM HERE
    if os.path.exists(os.path.join(responses_tmp_dir, "llm_response_3.json")):
        with open(os.path.join(responses_tmp_dir, "llm_response_3.json"), "r") as f:
            final_constraints_and_dynamic_rewards = json.load(f)
    else:
        if use_gemini or use_openai:
          llm_response_3 = call_llm(use_gemini, use_openai, llm_instruction_3, llm_user_prompt_3)
          final_constraints_and_dynamic_rewards = json.loads(llm_response_3.split("```json")[1].split("```")[0].strip())
        else:
          llm_response_3 = None
          final_constraints_and_dynamic_rewards = None
        if final_constraints_and_dynamic_rewards is not None:
            with open(os.path.join(responses_tmp_dir, "llm_response_3.json"), "w") as f:
                json.dump(final_constraints_and_dynamic_rewards, f)
        print(f"[Ashok] Saved final constraints and dynamic rewards to {os.path.join(responses_tmp_dir, 'llm_response_3.json')}")
            
    saved_final_constraints_and_dynamic_rewards = save_reward_functions(final_constraints_and_dynamic_rewards, reward_code_dir=f"{user_query}_dynamic_reward_functions_final")
    if not saved_final_constraints_and_dynamic_rewards:
        raise ValueError("Failed to save final constraints and dynamic rewards")
      
    verified_tests = verify_tests_for_reward_function(room_type, user_query, reward_code_dir=f"{user_query}_dynamic_reward_functions_final")
    if not verified_tests:
        raise ValueError("Failed to verify tests for final constraints and dynamic rewards")
    print("Tests verified for final constraints and dynamic rewards")
    
    # llm_instruction_4, llm_user_prompt_4 = create_prompt_4(user_prompt, final_constraints_and_dynamic_rewards, room_type)
    # with open(os.path.join(prompts_tmp_dir, "llm_instruction_4.md"), "w") as f:
    #     f.write(llm_instruction_4)
    # with open(os.path.join(prompts_tmp_dir, "llm_user_prompt_4.md"), "w") as f:
    #     f.write(llm_user_prompt_4)
        
    # # CALL THE LLM HERE
    # if os.path.exists(os.path.join(responses_tmp_dir, "llm_response_4.json")):
    #     with open(os.path.join(responses_tmp_dir, "llm_response_4.json"), "r") as f:
    #         reward_weights = json.load(f)
    # else:
    #     if use_gemini or use_openai:
    #       llm_response_4 = call_llm(use_gemini, use_openai, llm_instruction_4, llm_user_prompt_4)
    #       reward_weights = json.loads(llm_response_4.split("```json")[1].split("```")[0].strip())
    #     else:
    #       llm_response_4 = None
    #       reward_weights = None
    #     if reward_weights is not None:
    #         with open(os.path.join(responses_tmp_dir, "llm_response_4.json"), "w") as f:
    #             json.dump(reward_weights, f)
    #     print(f"[Ashok] Saved reward weights to {os.path.join(responses_tmp_dir, 'llm_response_4.json')}")
        
    # print(f"Reward weights: {reward_weights}")
    
    # print("Successfully completed the task")
    
def get_statistics_for_final_rewards(cfg, load):
    user_query = cfg.algorithm.ddpo.dynamic_constraint_rewards.user_query
    from dynamic_constraint_rewards.commons import get_reward_stats_from_dataset_helper, get_reward_stats_from_baseline_helper
    base_dir = os.path.dirname(__file__)
    reward_functions = json.load(open(os.path.join(base_dir, f"{user_query}_responses_tmp", "llm_response_3.json")))
    inpaint_masks = reward_functions["inpaint"]
    inpaint_masks = str(inpaint_masks).replace("'", '')
    print(inpaint_masks)
    print(type(inpaint_masks))
    threshold_dict = {}
    for reward in reward_functions["rewards"]:
        threshold_dict[reward["name"]] = reward["success_threshold"]
    print(threshold_dict)
    dataset_stats = get_reward_stats_from_dataset_helper(cfg, reward_code_dir=f"{user_query}_dynamic_reward_functions_final", threshold_dict=threshold_dict)
    baseline_stats = get_reward_stats_from_baseline_helper(cfg, load=load, inpaint_masks=inpaint_masks, threshold_dict=threshold_dict, reward_code_dir=f"{user_query}_dynamic_reward_functions_final")  
    print(f"Dataset stats: {dataset_stats}")
    print(f"Baseline stats: {baseline_stats}")

    return dataset_stats, baseline_stats

    

@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig):
    register_resolvers()
    OmegaConf.resolve(cfg)
    
    # USER_PROMPT = "A classroom for 10 students."
    # USER_PROMPT = "A bedroom with ceiling lamp above each corner of the bed."
    USER_PROMPT = "I want to follow Vaastu for bedroom layout. The bed's headboard should face east."
    
    # Bedroom
    run_llm_pipeline(cfg.dataset.data.room_type, cfg, use_openai=False, get_stats=False, load="fhfnf4xi")
    # get_statistics_for_final_rewards(cfg, load="fhfnf4xi")
    
    
    # Livingroom
    # run_llm_pipeline(USER_PROMPT, cfg.dataset.data.room_type, cfg, use_openai=False, get_stats=False, load="cu8sru1y")
    # get_statistics_for_final_rewards(cfg, load="w0gmpwep")
    
if __name__ == "__main__":
    main()
    
# python dynamic_constraint_rewards/run_llm_pipeline.py dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150 dataset.data.room_type=bedroom dataset.model_path_vec_len=30 dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData dataset.data.dataset_directory=bedroom dataset.data.annotation_file=bedroom_threed_front_splits_original.csv dataset.max_num_objects_per_scene=12 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=30 algorithm.custom.obj_diff_vec_len=30 algorithm.custom.num_classes=22 dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm algorithm.validation.num_samples_to_render=0 algorithm.validation.num_samples_to_visualize=0 algorithm.validation.num_directives_to_generate=0 algorithm.test.num_samples_to_render=0 algorithm.test.num_samples_to_visualize=0 algorithm.test.num_directives_to_generate=0 algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 algorithm.custom.old=True algorithm.ddpo.dynamic_constraint_rewards.user_query="I want to follow Vaastu for bedroom layout. The bed's headboard should face east." algorithm.ddpo.dynamic_constraint_rewards.use=True
    
# python dynamic_constraint_rewards/run_llm_pipeline.py dataset=custom_scene dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json dataset._name=custom_scene +num_scenes=1000 algorithm=scene_diffuser_midiffusion algorithm.trainer=ddpm experiment.find_unused_parameters=True algorithm.classifier_free_guidance.use=False algorithm.classifier_free_guidance.use_floor=True algorithm.classifier_free_guidance.weight=0 algorithm.custom.loss=true algorithm.ema.use=True algorithm.noise_schedule.scheduler=ddim algorithm.noise_schedule.ddim.num_inference_timesteps=150  dataset.data.room_type=livingroom dataset.model_path_vec_len=30 dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/MiData dataset.data.dataset_directory=livingroom dataset.data.annotation_file=livingroom_threed_front_splits.csv dataset.max_num_objects_per_scene=21 algorithm.custom.objfeat_dim=0 algorithm.custom.obj_vec_len=65 algorithm.custom.obj_diff_vec_len=65 algorithm.custom.num_classes=25 dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm algorithm.validation.num_samples_to_render=0 algorithm.validation.num_samples_to_visualize=0 algorithm.validation.num_directives_to_generate=0 algorithm.test.num_samples_to_render=0 algorithm.test.num_samples_to_visualize=0 algorithm.test.num_directives_to_generate=0 algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0