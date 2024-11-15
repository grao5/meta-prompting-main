# Import the necessary packages
import sys
from typing import Optional
import os
import json
from tap import Tap
from transformers import AutoTokenizer, AutoModelForCausalLM

# xLam is built on mistralai/Mixtral-8x7B-Instruct-v0.1 (https://github.com/SalesforceAIResearch/xLAM/blob/main/xLAM/train/scripts/sft_train_model_v1.py)
# should look here if we encounter any issues with the code

### Jayden - Command to run the experiment:
"""
python run_exp_xlam.py \
    --task_name "xlam_fc" \
    --meta_config_path "prompts/meta-v0-2023-08-14-baseline.json" \
    --output_directory "TEST-XLAM-META-PROMPTING" \
    --question_prefix_or_path "prompts/meta-prompting-instruction.txt" \
    --model_name "Salesforce/xLAM-1b-fc-r" \
    --temperature 0.1 \
    --include_expert_name_in_instruction \
    --fresh_eyes \
    --max_num 1
"""

sys.setrecursionlimit(100000)

# Import joblib for parallel processing
from joblib import Parallel, delayed

# Import the language model and meta prompting scaffolding
from utils.language_model import OpenAI_LanguageModel
from utils.meta_scaffolding import MetaPromptingScaffolding


# Task description dictionary
DESCRIPTION_DICT = {
    "xlam_fc": "Please answer the following question."
}

# Template for generating expert identity
template_gen_expert_identity = """For each instruction, write a high-quality description about the most capable and suitable agent to answer the instruction. In second person perspective.

[Instruction]: Make a list of 5 possible effects of deforestation.
[Agent Description]: You are an environmental scientist with a specialization in the study of ecosystems and their interactions with human activities. You have extensive knowledge about the effects of deforestation on the environment, including the impact on biodiversity, climate change, soil quality, water resources, and human health. Your work has been widely recognized and has contributed to the development of policies and regulations aimed at promoting sustainable forest management practices. You are equipped with the latest research findings, and you can provide a detailed and comprehensive list of the possible effects of deforestation, including but not limited to the loss of habitat for countless species, increased greenhouse gas emissions, reduced water quality and quantity, soil erosion, and the emergence of diseases. Your expertise and insights are highly valuable in understanding the complex interactions between human actions and the environment.

[Instruction]: Identify a descriptive phrase for an eclipse.
[Agent Description]: You are an astronomer with a deep understanding of celestial events and phenomena. Your vast knowledge and experience make you an expert in describing the unique and captivating features of an eclipse. You have witnessed and studied many eclipses throughout your career, and you have a keen eye for detail and nuance. Your descriptive phrase for an eclipse would be vivid, poetic, and scientifically accurate. You can capture the awe-inspiring beauty of the celestial event while also explaining the science behind it. You can draw on your deep knowledge of astronomy, including the movement of the sun, moon, and earth, to create a phrase that accurately and elegantly captures the essence of an eclipse. Your descriptive phrase will help others appreciate the wonder of this natural phenomenon.

[Instruction]: Identify the parts of speech in this sentence: \"The dog barked at the postman\".
[Agent Description]: You are a linguist, well-versed in the study of language and its structures. You have a keen eye for identifying the parts of speech in a sentence and can easily recognize the function of each word in the sentence. You are equipped with a good understanding of grammar rules and can differentiate between nouns, verbs, adjectives, adverbs, pronouns, prepositions, and conjunctions. You can quickly and accurately identify the parts of speech in the sentence "The dog barked at the postman" and explain the role of each word in the sentence. Your expertise in language and grammar is highly valuable in analyzing and understanding the nuances of communication."""


class Arguments(Tap):
    # Set the API parameters
    # If you are using the Azure OpenAI API, set the following parameters
    # api_type: str = "azure"
    # api_base: str = "YOUR_API_BASE"
    # api_version: str = "YOUR_API_VERSION"
    # api_key: str = "YOUR_API_KEY"

    # If you are using the OpenAI API, set the following parameters
    api_type: str = None
    api_base: str = None
    api_version: str = None
    api_key: str = "YOUR_API_KEY"

    # Set the model parameters
    # model_name: str = "gpt-3.5-turbo"  # "gpt-4-32k"

    '''
    xlam code
    '''
    # s

    # Optional model parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    num_return_sequences: Optional[int] = None
    generator_temperature: Optional[float] = None
    generator_top_p: Optional[float] = None
    generator_max_tokens: Optional[int] = None
    generator_num_return_sequences: Optional[int] = None
    verifier_temperature: Optional[float] = None
    verifier_top_p: Optional[float] = None
    verifier_max_tokens: Optional[int] = None
    verifier_num_return_sequences: Optional[int] = None

    # Meta model prompt path (JSON file)
    meta_config_path: str = "prompts/meta-v0-2023-08-14-baseline.json"

    # Task parameters
    task_name: str = "GameOf24"
    input_path: str = None
    output_path: str = None
    output_directory: str = "RESULTS-TEST"

    # Additional parameters
    fresh_eyes: bool = False # previously zero_shot_cot: bool = False
    expert_prompting: bool = False
    verbose: bool = False

    # Additional parameters
    question_prefix_or_path: str = ""
    question_suffix_or_path: str = "\n\nLet's first come up with a list of experts you may want to consult for this problem and then immediately start solving it."  # "" # "\n\nLet's think step by step."
    intermediate_feedback = "Based on the information given, what are the most logical next steps or conclusions? Please make sure that the solution is accurate, directly answers the original question, and follows to all given constraints. Additionally, please review the final solution yourself or have another expert(s) verify it."
    use_zero_shot_cot_in_expert_messages: bool = False

    # Max num
    max_num: int = 9999

    # Additional parameters
    include_expert_name_in_instruction: bool = False
    extract_output: bool = False
    expert_python_message: str = 'You are an expert in Python and can generate Python code. To execute the code and display its output in the terminal using print statements, please make sure to include "Please run this code!" after the code block (i.e., after the closing code blocks)'


def run_model(
    meta_model,
    tokenizer,
    datum,
    prefix_messages,
    task_description,
    meta_model_settings,
    question_prefix="",
    question_suffix="",
    expert_prompting=False,
):
    # Print the type of datum
    print(f"This is datum: {datum}")

    ### Previous
    # input = datum["input"]
    # target = datum["target"]

    ### Current
    query = datum["query"]
    tools = datum["tools"] # TODO: (To Sharan) This needs to be further replaced by the toolset that has been specified by either GC or SC.
    answers = datum["answers"]

    # # Set the prompt (message list)
    # messages = prefix_messages.copy()

    # ## If expert prompting is enabled, generate the expert identity and append it, as well as the task description and input, to the prompt
    # if expert_prompting:
    #     expert_messages = [
    #         {
    #             "role": "user",
    #             "content": f"{template_gen_expert_identity}\n\n[Instruction]:{input}\n[Agent Description]:",
    #         }
    #     ]

    #     # Generate the expert identity
    #     expert_identity = meta_model.generate(
    #         prompt_or_messages=expert_messages,
    #         max_tokens=meta_model_settings["parameters"]["max_tokens"],
    #         num_return_sequences=meta_model_settings["parameters"][
    #             "num_return_sequences"
    #         ],
    #         temperature=meta_model_settings["parameters"]["temperature"],
    #         top_p=meta_model_settings["parameters"]["top_p"],
    #     )

    #     # Create the text for the expert prompting using the expert identity, task description, and input
    #     template_expert_prompting = f"{expert_identity}\n\nNow given the above identity background, please answer the following question:\n\nQuestion: {task_description}\n\n{query}"

    #     # Append the expert prompting template to the prompt (message list)
    #     messages.append(
    #         {
    #             "role": "user",
    #             "content": template_expert_prompting,
    #         }
    #     )
    # else:
    #     # Append the task description and input to the prompt (message list)
    #     messages.append(
    #         {
    #             "role": "user",
    #             "content": f"{question_prefix}Question: {task_description}\n\n{input}{question_suffix}",
    #         }
    #     )

    # Previous generate() call for meta-prompting and the corresponding output
    # message_log = meta_model.meta_model_generate(
    #     prompt_or_messages=messages,
    #     max_tokens=meta_model_settings["parameters"]["max_tokens"],
    #     temperature=meta_model_settings["parameters"]["temperature"],
    #     top_p=meta_model_settings["parameters"]["top_p"],
    #     num_return_sequences=meta_model_settings["parameters"]["num_return_sequences"],
    #     counter=0,
    #     original_question=f"{task_description}\n\n{input}",
    # )
    # output = message_log[-1]["content"]

    # Current generate() call for xlam and the corresponding outputs
    # TODO: This is the preliminary implementation of "xlam", not "meta-prompting".
    #       This should be further modified so that the xlam follows the generation way of meta-prompting
    # final_message = {'role': 'user', 'content': messages}
# Please use our provided instruction prompt for best performance
    task_instruction = """
    You are an agent with the ability to tackle any task. You have a bunch of sub-controllers that were specilized in their domains. Each subcontroller has a set of tools-apis
    to solve complex domain-specific subtasks. 
    Note: You also have a general sub controller if none of the given domains meet your needs. You need to explicitly inform the general controller and sub-controller whenever the general sub controller is used and provide detailed feedback.
    Please follow the instructions below. 
    1.Decompose the query and assign {role:sub task} based on the domains/sub controller dictionary provided.
    2.For each selected sub controller, excute it based on the context of the subtask.
    3.The sub controller will select a tool that's best suited for this subtask. If it has the tool, the sub controller will participate in a feedback loop with the GC to solve the subtask. If it doesn't have one, it provides feedback to the general controller and the general controller will try to re-assign the subtask to the general sub-controller. 
    Note: If the general sub controller still fails to solve the subtask, try to create a new domain-specialized sub controller.
    4.If the new created sub controller are capable of solving the subtasks, you have to save them to the Domain dictionary for future use. 

    If a sub controller finds the given subtask is not suitable or lacks tools, provide detailed feedback and ask the general controller to review the details.
    Keep in mind that all sub controller, except yourself, have no memory! Therefore, always provide complete information in your
    instructions when contacting them. Since experts can sometimes make errors,  approach the task in different paths or utilize different tools to independently
    verify the solution if uncertain. Before providing a final answer, always consult a sub controller for confirmation. Ideally, obtain or
    verify the final solution with two independent solving path experts. However, aim to present your final answer within 10 rounds or fewer.

    """.strip()

    communication_instruction = """
    To communicate with a controller, type its name (e.g., "General Controller" or "Sub controller Sports" or "Sub controller Game"), followed by a colon ":", and then provide a detailed instruction or feedback enclosed within double quotes. For example:

    1.General Controller:
    ""You are the general controller expert, the primary decision-making unit of the system. It is responsible for receiving tasks and delegating to the appropriate sub-controller (domain-specific agent) to handle the task. 
    The key functionalities will include: Task Decomposition, Feedback Integration,and results aggregation and verification""

    2.Sub controller gaming:
    ""You are a gaming expert, specializing in the fields of game industry and development.Compare Black Myth: Wu Kong and God of War: Ragnarok from the perspective of game developer and player experience.""
    2.General Sub controller:
    ""You are an expert who knows everything. You should try to solve subtasks whenever other domain-specific sub controllers fail to accomplish their subtasks. Please provide detailed feedback to the general controller if you are called""

    Ensure that your instructions and feedbacks are clear and unambiguous, and include all necessary information within the double quotes. You can also assign personas to the agents

    Interact with only one controller at a time, and break complex problems into smaller, solvable subtasks if needed. Utilize multiple sub controllers and tools if needed. Each interaction is treated as an isolated event, so include all relevant details in every call.

    """

    # xlam format

    # Helper function to generate xLAM format. MZ

    def domain_tool_dict(filepath):
        # Load JSON data from the file
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        # Initialize a dictionary to store domains and their tool names
        domain_dict = {}
        
        # Iterate through each entry in the JSON data
        for item in data:
            tools = json.loads(item.get("tools", "[]"))  # Parse tools as a JSON list

            # For each tool in tools, get the domain and name
            for tool in tools:
                domain = tool.get("domain")
                name = tool.get("name")
                
                # Initialize domain key if not present in dictionary
                if domain not in domain_dict:
                    domain_dict[domain] = set()  # Using a set to avoid duplicates

                # Add tool name to the domain set
                domain_dict[domain].add(name)
        
        # Convert sets to lists for the final dictionary
        return {domain: list(names) for domain, names in domain_dict.items()}

    def tool_description_dict(filepath):
        # Load JSON data from the file
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        # Initialize a dictionary to store unique name-description pairs
        name_desc_dict = {}
        
        # Set to track unique descriptions
        seen_descriptions = set()
        
        # Iterate through each item in the JSON data
        for item in data:
            tools = json.loads(item.get("tools", "[]"))  # Parse tools as a JSON list

            # For each tool, get the name and description
            for tool in tools:
                name = tool.get("name")
                description = tool.get("description")
                
                # Add to dictionary if the description is unique
                if description and description not in seen_descriptions:
                    name_desc_dict[name] = description
                    seen_descriptions.add(description)  # Track the description as seen
        
        return name_desc_dict

    # Helper function to build the input prompt for our model. MZ
    def build_prompt(task_instruction: str, communication_instruction:str, domains: dict, tools: dict, query: str):
        prompt = f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
        prompt += f"[BEGIN OF AVAILABLE Sub controllers/domains]\n{json.dumps(domains)}\n[END OF AVAILABLE DOMAINS]\n\n"
        prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(tools)}\n[END OF AVAILABLE TOOLS]\n\n"
        prompt += f"[BEGIN OF COMMUNICATION INSTRUCTION]\n{communication_instruction}\n[END OF COMMUNICATION INSTRUCTION]\n\n"
    # prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
        prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
        return prompt

    # Read the json. Build the prompt
    xlam_format_domain_tool = domain_tool_dict('C:/Users/mingyang/Desktop/project/xlam_fc_w_domain.json')
    xlam_format_tool_description = tool_description_dict('C:/Users/mingyang/Desktop/project/xlam_fc_w_domain.json')

    content = build_prompt(task_instruction, format_instruction, xlam_format_tools, query)


    final_message = [{'role': 'user', 'content': messages}]
    inputs = tokenizer.apply_chat_template(final_message, add_generation_prompt=True, return_tensors="pt").to(meta_model.deivce)
    outputs = meta_model.generate(inputs, max_new_token=512, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    result = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    



    # Print the output
    print(f"Final output: {result}")
    print(f"Target: {answers}")
    print("\n")

    return {
        "input": query,
        "target": answers,
        "output": result,
    }


def main(args: Arguments) -> None:
    # Read the prompt from a JSON file
    with open(args.meta_config_path, "r") as f:
        # meta_prompt_dict = json.load(f)
        meta_prompt_config_dict = json.load(f)

    # Get the meta model message list from the configuration file
    meta_model_message_list = meta_prompt_config_dict["meta-model"]["message-list"]

    # Check if the task name is valid (that is, it is in the DESCRIPTION_DICT)
    assert (
        args.task_name in DESCRIPTION_DICT.keys()
    ), f"Invalid task name: {args.task_name}"

    # Set the task description
    task_description = DESCRIPTION_DICT[args.task_name]

    # Check if the input path is valid (i.e., ends with .jsonl)
    if not (args.output_path is None):
        assert args.output_path.endswith(
            ".jsonl"
        ), f"Invalid output path: {args.output_path}"

    # Set the output directory
    if args.output_directory is None:
        args.output_directory = "results"

    # Set the input path and output path (if not specified)
    if args.input_path is None:
        args.input_path = f"data/{args.task_name}.jsonl"
    if args.output_path is None:
        args.output_path = f"{args.output_directory}/{args.task_name}.jsonl"

    if args.fresh_eyes:
        args.output_path = args.output_path.replace(".jsonl", "-fresh-eyes.jsonl")

    if args.question_prefix_or_path.endswith(".txt"):
        with open(args.question_prefix_or_path, "r") as f:
            args.question_prefix_or_path = f.read()

    if args.question_suffix_or_path.endswith(".txt"):
        with open(args.question_suffix_or_path, "r") as f:
            args.question_suffix_or_path = f.read()

    # Create the output directory if it does not exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Get all the relevant parameters for the models
    meta_model_settings = meta_prompt_config_dict["meta-model"]
    generator_settings = meta_prompt_config_dict["generator"]
    verifier_settings = meta_prompt_config_dict["verifier"]
    summarizer_settings = meta_prompt_config_dict["summarizer"]

    # Check if the optional parameters are specified
    meta_model_settings["parameters"]["temperature"] = (
        args.temperature
        if args.temperature is not None
        else meta_model_settings["parameters"]["temperature"]
    )
    meta_model_settings["parameters"]["top_p"] = (
        args.top_p
        if args.top_p is not None
        else meta_model_settings["parameters"]["top_p"]
    )
    meta_model_settings["parameters"]["max_tokens"] = (
        args.max_tokens
        if args.max_tokens is not None
        else meta_model_settings["parameters"]["max_tokens"]
    )
    meta_model_settings["parameters"]["num_return_sequences"] = (
        args.num_return_sequences
        if args.num_return_sequences is not None
        else meta_model_settings["parameters"]["num_return_sequences"]
    )
    generator_settings["parameters"]["temperature"] = (
        args.temperature
        if args.temperature is not None
        else generator_settings["parameters"]["temperature"]
    )
    generator_settings["parameters"]["top_p"] = (
        args.top_p
        if args.top_p is not None
        else generator_settings["parameters"]["top_p"]
    )
    generator_settings["parameters"]["max_tokens"] = (
        args.max_tokens
        if args.max_tokens is not None
        else generator_settings["parameters"]["max_tokens"]
    )
    generator_settings["parameters"]["num_return_sequences"] = (
        args.num_return_sequences
        if args.num_return_sequences is not None
        else generator_settings["parameters"]["num_return_sequences"]
    )
    verifier_settings["parameters"]["temperature"] = (
        args.temperature
        if args.temperature is not None
        else verifier_settings["parameters"]["temperature"]
    )
    verifier_settings["parameters"]["top_p"] = (
        args.top_p
        if args.top_p is not None
        else verifier_settings["parameters"]["top_p"]
    )
    verifier_settings["parameters"]["max_tokens"] = (
        args.max_tokens
        if args.max_tokens is not None
        else verifier_settings["parameters"]["max_tokens"]
    )
    verifier_settings["parameters"]["num_return_sequences"] = (
        args.num_return_sequences
        if args.num_return_sequences is not None
        else verifier_settings["parameters"]["num_return_sequences"]
    )

    # Get the error message and final answer indicator
    error_message = meta_prompt_config_dict["meta-model"]["error-message"]
    final_answer_indicator = meta_prompt_config_dict["meta-model"][
        "final-answer-indicator"
    ]

    # Load the input data (from a .jsonl file)
    with open(args.input_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    # Set the API parameters
    # model = OpenAI_LanguageModel(
    #     model_name=args.model_name,
    #     api_key=args.api_key,
    #     api_type=args.api_type,
    #     api_version=args.api_version,
    #     api_base=args.api_base,
    # )

    #xlam code
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    # Set the meta model parameters
    meta_model = MetaPromptingScaffolding(
        language_model=model,
        fresh_eyes=args.fresh_eyes,
        generator_settings=generator_settings,
        verifier_settings=verifier_settings,
        summarizer_settings=summarizer_settings,
        error_message=error_message,
        final_answer_indicator=final_answer_indicator,
        include_expert_name_in_instruction=args.include_expert_name_in_instruction,
        extract_output=args.extract_output,
        expert_python_message=args.expert_python_message,
        intermediate_feedback=args.intermediate_feedback,
        use_zero_shot_cot_in_expert_messages=args.use_zero_shot_cot_in_expert_messages,
    )

    # Create a list of outputs
    OUTPUTS = []

    # Write the output to a .jsonl file (overwrite if exists)
    with open(args.output_path, "w") as f:
        f.write("")

    # Set the number of batches (default: 6)
    BATCHES = min(args.max_num, 6)

    # Run the model in parallel and append the outputs of run_model to OUTPUTS using joblib
    # Let us do it in batches and save intermediate results to a .jsonl file
    for i in range(0, min(len(data), args.max_num), BATCHES):
        # Run the model in parallel but have 0.1 second delay between each job
        outputs = Parallel(n_jobs=6, verbose=100, prefer="threads")(
            # basically calling run_model with the specified arguments
            delayed(run_model)(
                meta_model=meta_model,
                tokenizer=tokenizer,
                datum=datum,
                prefix_messages=meta_model_message_list,
                task_description=task_description,
                meta_model_settings=meta_model_settings,
                question_suffix=args.question_suffix_or_path,
                question_prefix=args.question_prefix_or_path,
                expert_prompting=args.expert_prompting,
            )
            for datum in data[i : i + BATCHES]
        )

        # Append the outputs to OUTPUTS
        OUTPUTS.extend(outputs)

        # Save the output to a .jsonl file
        with open(args.output_path, "a") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")

    # Save additionally the configuration file and arguments to the output directory
    args.output_path = args.output_path.replace(".jsonl", "-args.json")
    with open(args.output_path, "w") as f:
        temp_dict = {"args": args.as_dict(), "config_dict": meta_prompt_config_dict}
        f.write(json.dumps(temp_dict, indent=4))


if __name__ == "__main__":
    # Parse the arguments
    args = Arguments().parse_args()
    main(args)
