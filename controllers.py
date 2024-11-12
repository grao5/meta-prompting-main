import json
import xlam_cpp  # Salesforce/xLAM-1b-fc-r API wrapper for LLM queries
from collections import defaultdict

# Populate Registries from tools_general.json
# External Registries (Global)
def populate_registries(toolset_path="tools_general.json"):
    """
    Populates the toolset and sub-controllers registry from the tools_general.json dataset.
    """
    toolset = {}
    sub_controllers_registry = defaultdict(list)

    try:
        # Load the tools_general.json file
        with open(toolset_path, "r") as file:
            tools_data = json.load(file)

        # Populate toolset and sub_controllers_registry
        for tool in tools_data:
            tool_name = tool["name"]
            domain = tool.get("domain", "Other")

            # If the domain is "Other", add the tool to the "General" domain
            if domain == "Other":
                toolset[tool_name] = {
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                    "domain": "General"
                }
                sub_controllers_registry["General"].append(tool)
            else:
                toolset[tool_name] = {
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                    "domain": domain
                }
                sub_controllers_registry[domain].append(tool)

    except Exception as e:
        print(f"Error while populating registries: {e}")

    return toolset, dict(sub_controllers_registry)

toolset, sub_controllers_registry = populate_registries(toolset_path="/path/to/tools_general.json")

# GeneralController Class
class GeneralController:
    def __init__(self, sub_controllers_registry, toolset, llm_api_key, max_feedback_loops=3):
        self.sub_controllers_registry = sub_controllers_registry  # Registry of Sub-Controller domains and tools
        self.toolset = toolset  # Registry of tools with their descriptions and parameters
        self.llm_api_key = llm_api_key
        self.max_feedback_loops = max_feedback_loops
        self.plugins = {}  # Storage for dynamically registered plug-ins

        # Initialize General Sub-Controller (GSC) if available in the registry or separately
        if "General" in sub_controllers_registry:
            general_tools = [toolset[tool_name] for tool_name in sub_controllers_registry["General"]]
            self.gsc = GeneralSubController(role="General", tools=general_tools)
        else:
            self.gsc = GeneralSubController()

        # Copy of permanent registries for temporary use during task lifecycle
        self.task_sub_controllers = sub_controllers_registry.copy()
        self.task_tools = toolset.copy()

        # Task Status Board to track progress of tasks assigned to different SCs
        self.task_status_board = {}

    def process_query(self, query):
        # Decompose the query into subtasks with dynamic role assignment
        tasks = self.decompose_query(query)
        responses = {}

        # Execute each subtask and provide feedback if necessary
        for role, task in tasks.items():
            if role in self.sub_controllers_registry:
                responses[role] = self.execute_with_feedback(role, task)
            else:
                # Attempt to handle with the General Sub-Controller (GSC) first
                gsc_response = self.execute_with_feedback("General", task)
                if gsc_response == "Error: Unknown role":
                    # Verify if we need to create a new SC
                    create_new_role = self.verify_creation(role, task)
                    if create_new_role:
                        responses[role] = self.create_new_sc(role, task)
                    else:
                        # Attempt to reassign to a suitable SC
                        reassigned_role = self.reassign_task(role, task)
                        if reassigned_role:
                            responses[reassigned_role] = self.execute_with_feedback(reassigned_role, task)
                        else:
                            # Fall back to the GSC if no suitable SC found
                            responses["General"] = self.execute_with_feedback("General", task)
                else:
                    responses[role] = gsc_response

        # Aggregate responses from all subtasks and return the final result
        final_response = self.aggregate_responses(query, responses)
        self.store_persistent_data(query, final_response, responses)
        return final_response

    def verify_creation(self, role, task):
        # Ask the LLM whether to proceed with creating a new SC or reassign
        prompt = (
            f"A role '{role}' was assigned to the following task, but no existing sub-controller matches this domain.\n"
            f"Task: {task}\n Verify if a new sub-controller should be created for this role, or can the task be assigned to an existing domain?"
            f"Available domains: {list(self.sub_controllers_registry.keys())}\n"
            f"return your answer only as either 1 or 0 to pass as boolean logic in an elif statement"
        )
        response = self.llm_query(prompt)
        return response
    
    def reassign_task(self, role, task):
        # Ask the LLM to suggest the most suitable existing sub-controller for the task
        prompt = (
            f"The task '{task}' could not be handled by the {role} sub-controller and needs to be re-assigned.\n"
            f"Consider the requirments of the task and the tools and domain area of the sub controllers to pick a new role"
            f"Available domains:\n{self.task_sub_controllers}\n"
            "Which domain is the best match for this task? Return just the domain name from the list"
        )
        response = self.llm_query(prompt)
        suggested_role = response.strip()
        if suggested_role in self.sub_controllers_registry:
            return suggested_role
        return None

    def register_plugin(self, plugin_name, plugin_instance):
        """Registers a new external plug-in sub-controller"""
        self.plugins[plugin_name] = plugin_instance
        print(f"Plug-in {plugin_name} has been registered successfully.")

    def decompose_query(self, query):
        # Use LLM to break down the query into subtasks and assign appropriate roles
        prompt = (
            f"Understand the requirements of the input query \nQuery: {query}\n"
            f"and decompose it into subtasks that can be handled by the available domain experts" 
            f"based on their capabilities and tools available to them:"
            f"\n{self.task_sub_controllers}\n"
            "For each subtask, using a consistent requirements based scoring criteria, provide a score (0-10)"
            "to rank which domain is best suited. You output should be in the form domain:score for the top 10 domains"
            "so that the following line of code can parse the response: role, score, task = line.split(': ', 2)"
        )
        response = self.llm_query(prompt)

        tasks = {}
        for line in response.splitlines():
            role, score, task = line.split(': ', 2)
            score = int(score)
            if score >= 8:  # Threshold for matching an SC
                tasks[role] = task
            else:
                tasks["General"] = task  # Assign to GSC if no good match is found

        # Update the Task Status Board with initial statuses
        for role in tasks.keys():
            self.task_status_board[role] = "Pending"

        return tasks

    def execute_with_feedback(self, role, task):
        feedback_list = []  # List to hold feedback history
        sc = None

        # Initialize the appropriate SC if not already present
        if role == "General":
            sc = self.gsc
        elif role in self.sub_controllers_registry:
            sc = self.initialize_sc(role)
        else:
            return "Error: Unknown role."

        # Feedback loop for task execution
        for _ in range(self.max_feedback_loops):
            result = sc.handle_task(task, feedback=feedback_list)
            if self.verify_result(task, result):
                self.task_status_board[role] = "Completed"
                return result
            # Generate feedback and append to the feedback list
            feedback = self.generate_feedback(task, result)
            feedback_list.append(feedback)

            if "converged" in feedback.lower():
                break

        self.task_status_board[role] = "Failed to Converge"
        return result

    def verify_result(self, task, result):
    # Verify quality of result using LLM
    prompt = (
        f"Verification Agent: You are responsible for evaluating the accuracy and quality of the provided result.\n"
        f"Task: {task}\n and Task Result: {result}\n"
        f"Analyze the result carefully, and determine whether it meets the requirements fully, partially, or not at all.\n"
        f"Make sure to consider correctness, completeness, and alignment with the intended task.\n"
        "Your response should be either 'True' if the result meets the requirements or 'False' if it does not.\n"
        "Strictly provide your response in one word: 'True' or 'False'."
    )
    verification_response = self.llm_query(prompt)
    return verification_response.lower() == "true"


    def generate_feedback(self, task, result):
        # Generate feedback for improving the result
        prompt = (
          f"Provide detailed feedback on how to improve the following result.\nTask: {task}\nResult: {result}"
          "If you feel that no more feedback is needed, return just the key word 'converged' and nothing else"
        )
        return self.llm_query(prompt)

    def create_new_sc(self, role, task):
        # Create a new sub-controller dynamically for a new role
        prompt = (
            f"Create a new sub-controller for domain '{role}' to handle this task.\n"
            f"Take reference from existing sub-controllers to provide comprehensive instructions."
        )
        sc_instructions = self.llm_query(prompt)
        new_sc = DynamicSubController(role, sc_instructions)
        self.task_sub_controllers[role] = f"Handles domain '{role}' tasks as per given instructions."
        return new_sc.handle_task(task)

    def store_persistent_data(self, query, final_response, responses):
        # Store successful SCs or tools into permanent registries
        for role, response in responses.items():
            if "success" in response.lower():
                if role not in sub_controllers_registry:
                    sub_controllers_registry[role] = self.task_sub_controllers[role]

    def aggregate_responses(self, query, responses):
        # Aggregate results meaningfully
        prompt = (
            "Various domain specialized sub-controllers have completed the tasks they were assigned and returned 
            f"the results as below for the query: \n{query}\n"
            f"Subtask results: {responses}\n"
            "Combine the subtasks' results into a meaningful response to answer the original query.\n"
        )
        return self.llm_query(prompt)

    def llm_query(self, prompt):
        # Make LLM API call using Salesforce/xLAM-1b-fc-r
        response = xlam_cpp.Completion.create(
            model="xlam-1b-fc-r",  # Adjust model if needed
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()

    def initialize_sc(self, role):
        """
        Initializes an existing sub-controller using the domain information from the sub_controllers_registry.
        """
        tools = [self.task_tools[tool_name] for tool_name in self.sub_controllers_registry[role]]
        return SubController(role=role, tools=tools)

# SubController Class
class SubController:
    def __init__(self, role, tools):
        self.role = role  # The domain or role this SubController is responsible for
        self.tools = tools  # A list of tools available for this domain
        self.capabilities = self.infer_capabilities()  # Infer capabilities based on tool descriptions

    def infer_capabilities(self):
        """
        Infers the capabilities of the SubController based on the descriptions of the tools it has available.
        """
        inferred_capabilities = set()

        # Iterate over each tool to extract and infer capabilities based on tool descriptions
        for tool in self.tools:
            description = tool.get('description', '')
            capabilities = description.split()  # Split description into words for inference
            inferred_capabilities.update(capabilities)

        # Join unique capabilities into a readable string
        return ', '.join(sorted(inferred_capabilities))

    def handle_task(self, task, feedback=[]):
        """
        Handles the given task by using available tools and domain expertise.
        Takes into consideration any feedback from the GeneralController during the task execution lifecycle.
        """

        # Formulate the prompt to be used with the LLM
        prompt = (
            f"As an expert in the '{self.role}' domain, you are tasked with handling the following request: '{task}'.\n"
            f"Your tools include: {', '.join([tool['name'] for tool in self.tools])}\n"
            f"Tool Specific Capabilities: {self.capabilities}\n"
            f"Feedback from previous iterations (if any): {', '.join(feedback)}\n"
            "Please use the tools available to you to provide the best possible response to the task requirements.\n"
            "If you need more information for tool parameters, clearly state what information youn need and why\n.
            "If you feel all the tools are insufficient, clearly state why and provide detailed instructions for additional tools needed."
        )

        # Make LLM API call using Salesforce/xLAM-1b-fc-r
        response = xlam_cpp.Completion.create(
            model="xlam-1b-fc-r",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        ).choices[0].text.strip()

        return response

# GeneralSubController Class
class GeneralSubController(SubController):
    def __init__(self, role="General", tools=None):
        """
        Initialize the General Sub-Controller. 
        This is a special Sub-Controller capable of handling tasks that do not fall under any specific domain.
        """
        if tools is None:
            tools = []  # GeneralSubController can have a set of tools as needed, defaulting to an empty list.
        
        super().__init__(role, tools)

    def handle_task(self, task, feedback=[]):
        """
        Handles general tasks that don't match specific sub-controller domains.
        """
        prompt = (
            f"As the general-purpose controller, you are handling the following task: '{task}'.\n"
            f"Your tools include: {', '.join([tool['name'] for tool in self.tools]) if self.tools else 'No specific tools available'}.\n"
            f"Tool Specific Capabilities: {self.capabilities}\n"
            f"Feedback from previous iterations (if any): {', '.join(feedback)}\n"
            "Please provide a response that addresses the requirements as comprehensively as possible.\n"
            "Please use the tools available to you to provide the best possible response to the task requirements.\n"
            "If you need more information for tool parameters, clearly state what information youn need and why\n.
            "If you feel all the tools are insufficient, clearly state why and provide detailed instructions for additional tools needed."
        )

        # Make LLM API call using Salesforce/xLAM-1b-fc-r
        response = xlam_cpp.Completion.create(
            model="xlam-1b-fc-r",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        ).choices[0].text.strip()

        return response

