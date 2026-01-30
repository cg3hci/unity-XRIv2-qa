import os
import re
import subprocess

ROOT_DIR = './Python/'  # Root directory of your project
CACHED_INSTALLED_DEPS_FILENAME = 'installed_deps.txt'
OUTPUT_FILENAME = 'real_requirements.txt'
BUILTIN_IN_PYTHON = []

def get_builtins_in_python():
    import sys
    import pkgutil

    builtin_modules = set(sys.builtin_module_names)

    # Identify the standard library directory (excluding site-packages)
    std_lib_dir = os.path.dirname(os.__file__)

    # Collect standard library modules excluding site-packages
    std_lib_modules = set(module.name for module in pkgutil.iter_modules([std_lib_dir]) if 'site-packages' not in module.module_finder.path)

    # Combine built-in and standard library modules
    return sorted(builtin_modules.union(std_lib_modules))
    

BUILTIN_IN_PYTHON = get_builtins_in_python()

#region Get Used Dependencies
def find_python_files(root_dir):
    python_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(subdir, file))
    return python_files

def extract_imports(file_path):
    imports = set()
    import_pattern = re.compile(r'^\s*(import|from)\s+([a-zA-Z_][\w\.]*)')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = import_pattern.match(line)
            if match:
                module = match.group(2).split('.')[0]  # Only get the root module
                imports.add(module)
    
    return imports

def filter_custom_modules(imports, root_dir):
    custom_modules = set()
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                module_name = os.path.splitext(file)[0]
                custom_modules.add(module_name)
    
    real_dependencies = (imports - custom_modules)
    real_dependencies -= set(BUILTIN_IN_PYTHON)
    return real_dependencies
#endregion

#region Get Installed Dependencies with Versions
def get_installed_dependencies_via_file():
    # Check if the file exists and is not empty
    FILE_EXISTS = os.path.exists(ROOT_DIR+CACHED_INSTALLED_DEPS_FILENAME)
    if not FILE_EXISTS:
        return get_installed_dependencies_via_conda()


    FILE_NOT_EMPTY = os.path.getsize(ROOT_DIR+CACHED_INSTALLED_DEPS_FILENAME) > 0
    if not FILE_NOT_EMPTY:
        return get_installed_dependencies_via_conda()

    # check if the structure is correct (i.e. each line is in the format package_name=version)
    FILE_STRUCTURE_CORRECT = True
    with open(ROOT_DIR+CACHED_INSTALLED_DEPS_FILENAME, 'r') as f:
        for line in f:
            if not line.startswith("#"):
                package_info = line.strip().split('=')
                if len(package_info) < 2:
                    FILE_STRUCTURE_CORRECT = False
                    break
    if not FILE_STRUCTURE_CORRECT:
        return get_installed_dependencies_via_conda()


    installed_deps = {}
    with open(ROOT_DIR+CACHED_INSTALLED_DEPS_FILENAME, 'r') as f:
        for line in f:
            if not line.startswith("#"):
                package_info = line.strip().split('=')
                if len(package_info) >= 2:
                    package_name = package_info[0]
                    package_version = package_info[2]
                    installed_deps[package_name] = package_version
    return installed_deps

def get_installed_dependencies_via_conda():
    installed_deps = {}

    # Execute the conda list --export command and capture the output
    result = subprocess.run(['conda', 'list', '--export'], stdout=subprocess.PIPE, text=True)

    # Process the output
    for line in result.stdout.splitlines():
        if not line.startswith("#"):
            package_info = line.strip().split('=')
            if len(package_info) >= 2:
                package_name = package_info[0]
                package_version = package_info[1]
                installed_deps[package_name] = package_version
    
    with open(ROOT_DIR+CACHED_INSTALLED_DEPS_FILENAME, 'w') as f:
        for dep, version in installed_deps.items():
            f.write(f"{dep}=={version}\n")

    return installed_deps
#endregion

def main():
    all_imports = set()
    
    python_files = find_python_files(ROOT_DIR)
    
    for file_path in python_files:
        imports = extract_imports(file_path)
        all_imports.update(imports)
    
    real_dependencies = filter_custom_modules(all_imports, ROOT_DIR)
    installed_deps = get_installed_dependencies_via_file()

    # Filter out only those dependencies that are installed
    # real_installed_deps = {dep: installed_deps[dep] for dep in real_dependencies if dep in installed_deps}
    real_nontrivial_deps = {}
    for dep in real_dependencies:
        if dep in installed_deps:
            real_nontrivial_deps[dep] = installed_deps[dep]
            print(f"{dep}=={installed_deps[dep]}")
        else:
            print(f"[WARN] {dep} is not installed. Maybe it is a standard library module?")

    # Optionally, save to a file
    with open(ROOT_DIR+OUTPUT_FILENAME, 'w') as f:
        for dep, version in sorted(real_nontrivial_deps.items()):
            f.write(f"{dep}=={version}\n")
            
if __name__ == "__main__":
    main()
