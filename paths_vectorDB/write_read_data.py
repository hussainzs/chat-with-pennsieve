from typing import List


def write_paths_and_descriptions_to_file(all_paths: List[str], all_descriptions: List[str]):
    """
    Write the formatted paths and descriptions to a file named vectordb_paths_descriptions.txt in the same directory.

    Args:
        all_paths (List[str]): List of Cypher paths.
        all_descriptions (List[str]): List of descriptions corresponding to the Cypher paths.
    """
    try:
        with open("vectordb_paths_descriptions.txt", "a", encoding="utf-8", errors="replace") as f:
            for idx, (path, description) in enumerate(zip(all_paths, all_descriptions), start=1):
                try:
                    f.write(f"Path: {path}\nDescription: {description}\n\n")
                except Exception as err:
                    print(f"Skipping path {idx} due to write error: {err}")
    except Exception as e:
        print(f"Error opening file vectordb_paths_descriptions.txt: {e}")


def read_paths_and_descriptions_from_file() -> List[List[str]]:
    """
    Read the paths and descriptions from a file named vectordb_paths_descriptions.txt and return them as a list of lists.

    Returns:
        List[List[str]]: A list containing two lists - one for paths and one for descriptions.
    """
    file_path = 'vectordb_paths_descriptions.txt'
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
