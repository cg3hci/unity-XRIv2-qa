import json
from typing import TypedDict
from typing import TypedDict, List, Dict, Any


# Define TypedDicts for the structure of the benchmark JSON
class QuestionAnswer(TypedDict):
    question: str
    answer: str

class Toolkit(TypedDict):
    name: str
    dataset: List[QuestionAnswer]

class Platform(TypedDict):
    name: str
    toolkits: List[Toolkit]

class BenchmarkInfo(TypedDict):
    name: str
    description: str
    version: str
    date: str
    author: str
    email: str

# Custom exception for validation errors
class BenchmarkValidationError(Exception):
    """Custom exception for benchmark validation errors."""
    pass

# Define the Benchmark class with methods
class Benchmark:
    def __init__(self, data: Dict[str, Any]):
        """
        Initializes the Benchmark class with data from a dictionary.
        
        Args:
            data (Dict[str, Any]): A dictionary containing benchmark data.
        """
        self.data = data
        self.benchmark_info: BenchmarkInfo = data.get('benchmark_info', {})
        self.platforms: List[Platform] = data.get('platforms', [])

        self._validate_data()

    def get_dataset(self, platform_name: str, toolkit_name: str) -> List[QuestionAnswer]:
        """
        Returns the dataset for a given platform and toolkit.
        
        Args:
            platform_name (str): The name of the platform.
            toolkit_name (str): The name of the toolkit.
        
        Returns:
            List[QuestionAnswer]: A list of question-answer dictionaries.
        
        Raises:
            BenchmarkValidationError: If the platform or toolkit is not found.
        """
        # Find the platform
        for platform in self.platforms:
            if platform['name'] == platform_name:
                # Find the toolkit within the platform
                for toolkit in platform['toolkits']:
                    if toolkit['name'] == toolkit_name:
                        return toolkit['dataset']
                raise BenchmarkValidationError(f"Toolkit '{toolkit_name}' not found in platform '{platform_name}'.")
        
        raise BenchmarkValidationError(f"Platform '{platform_name}' not found.")

    def get_benchmark_info(self) -> str:
        """Returns a formatted string with the benchmark information."""
        info = self.benchmark_info
        return (
            f"Benchmark Name: {info['name']}\n"
            f"Description: {info['description']}\n"
            f"Version: {info['version']}\n"
            f"Date: {info['date']}\n"
            f"Author: {info['author']}\n"
            f"Email: {info['email']}"
        )

    def print_all_data(self):
        """Prints all benchmark information, platforms, toolkits, and their datasets."""
        print(self.get_benchmark_info())
        for platform in self.platforms:
            print(f"Platform: {platform['name']}")
            for toolkit in platform["toolkits"]:
                print(f"  Toolkit: {toolkit['name']}")
                for qa in toolkit["dataset"]:
                    print(f"    Q: {qa['question']} A: {qa['answer']}")


    #region Validation Methods
    def _validate_data(self) -> bool:
        """
        Validates the benchmark data to ensure it follows the expected schema.
        
        Returns:
            bool: True if the data is valid, False otherwise.
        """
        # Validate benchmark_info
        self.__validate_benchmark_info(self.benchmark_info)
        
        # Validate platforms
        if not isinstance(self.platforms, list):
            raise BenchmarkValidationError("Invalid platforms: not a list")
        
        for platform in self.platforms:
            self.__validate_platform(platform)
        
        return True

    def __validate_benchmark_info(self, info: Dict[str, Any]) -> bool:
        """Validates the benchmark_info dictionary."""
        required_keys = {('name',str), ('description',str), ('version',str), ('date',str), ('author',str), ('email',str)}
        self.__validate_keys(info, required_keys, "benchmark_info")

    def __validate_platform(self, platform: Dict[str, Any]) -> bool:
        """Validates a platform dictionary."""
        
        required_keys = {('name',str), ('toolkits',list)}
        self.__validate_keys(platform, required_keys, "platform")

        for toolkit in platform['toolkits']:
            self.__validate_toolkit(toolkit)


    def __validate_toolkit(self, toolkit: Dict[str, Any]) -> bool:
        """Validates a toolkit dictionary."""
        required_keys = {('name',str), ('dataset',list)}
        self.__validate_keys(toolkit, required_keys, "toolkit")
        
        for qa in toolkit['dataset']:
            self.__validate_question_answer(qa)


    def __validate_question_answer(self, qa: Dict[str, Any]) -> bool:
        """Validates a question-answer pair dictionary."""
        required_keys = {('question',str), ('answer',str)}
        self.__validate_keys(qa, required_keys, "question-answer pair")

    def __validate_keys(self, data: Dict[str, Any], required_keys: set[tuple[str, type]], error_context: str) -> None:
        """
        Validates that the given dictionary contains all required keys with the correct types.
        
        Args:
            data (Dict[str, Any]): The dictionary to validate.
            required_keys (Set[Tuple[str, type]]): A set of tuples containing the required key and its expected type.
            error_context (str): A string to identify the context of the validation error message.
        
        Raises:
            BenchmarkValidationError: If a required key is missing or the type is incorrect.
        """
        for key, key_type in required_keys:
            if key not in data or not isinstance(data[key], key_type):
                raise BenchmarkValidationError(f"Invalid {error_context}: missing or invalid key '{key}'")
    #endregion


if __name__ == "__main__":
    # Load JSON data
    filename = "./benchmark.json"
    with open(filename, "r") as file:
        json_data = file.read()
    # Convert JSON string to a dictionary
    data_dict = json.loads(json_data)

    # Create the benchmark instance. It also internally checks if the structure is valid.
    benchmark = Benchmark(data_dict)
    # benchmark.print_all_data() # If you want to print all data

    # platform, toolkit = "Unity", "MRTK3-Mock"  # Example with real platform and mock toolkit
    # platform, toolkit = "WebMock", "A-Frame-Mock"  # Example with mock platform and toolkit
    platform, toolkit = "Unity", "XRIv2"

    dataset = benchmark.get_dataset(platform, toolkit)
    print(f"Dataset for Platform: {platform}, Toolkit: {toolkit}:\n{dataset}")