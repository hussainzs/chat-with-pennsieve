from typing import List


def write_paths_and_descriptions_to_file(all_paths: List[str], all_descriptions: List[str]):
    """
    Write the formatted paths and descriptions to a file named data.txt in the same directory.

    Args:
        all_paths (List[str]): List of Cypher paths.
        all_descriptions (List[str]): List of descriptions corresponding to the Cypher paths.
    """
    file_path = 'data.txt'
    try:
        with open(file_path, 'w') as file:
            for path, description in zip(all_paths, all_descriptions):
                file.write(f"Path: {path}\nDescription: {description}\n\n")
        print(f"Data successfully written to {file_path} 📄📄")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")


def read_paths_and_descriptions_from_file() -> List[List[str]]:
    """
    Read the paths and descriptions from a file named data.txt and return them as a list of lists.

    Returns:
        List[List[str]]: A list containing two lists - one for paths and one for descriptions.
    """
    file_path = 'data.txt'
    paths = []
    descriptions = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("Path:"):
                    current_path = line[len("Path: "):].strip()
                    paths.append(current_path)
                elif line.startswith("Description:"):
                    current_description = line[len("Description: "):].strip()
                    descriptions.append(current_description)
        return [paths, descriptions]
    except Exception as e:
        print(f"Error reading from file {file_path}: {e}")
        return [[], []]
