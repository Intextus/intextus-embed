import urllib.request
import json
import re
import sys

def get_latest_pypi_version(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'intextus-bump-script'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data["info"]["version"]
    except Exception as e:
        print(f"Could not retrieve version from PyPI (package might not exist yet): {e}")
        return None

def increment_version(version_str):
    if not version_str:
        return "0.1.0"
    
    # Split version by dots
    parts = version_str.split('.')
    try:
        # Increment the last part (patch version)
        parts[-1] = str(int(parts[-1]) + 1)
        return '.'.join(parts)
    except ValueError:
        return version_str + ".1"

def update_pyproject_version(new_version):
    file_path = "pyproject.toml"
    with open(file_path, "r") as f:
        content = f.read()
    
    # Find version = "..."
    pattern = r'(version\s*=\s*")([^"]+)(")'
    new_content, count = re.subn(pattern, rf'\g<1>{new_version}\g<3>', content)
    
    if count == 0:
        raise ValueError("Could not find version string in pyproject.toml")
        
    with open(file_path, "w") as f:
        f.write(new_content)
    print(f"Successfully updated pyproject.toml version to: {new_version}")

def main():
    package_name = "intextus-embed"
    current_pypi_version = get_latest_pypi_version(package_name)
    print(f"Current PyPI version: {current_pypi_version}")
    
    if not current_pypi_version:
        # Read from pyproject.toml to preserve initial version
        with open("pyproject.toml", "r") as f:
            content = f.read()
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            new_version = match.group(1)
        else:
            new_version = "0.1.0"
    else:
        new_version = increment_version(current_pypi_version)
        
    update_pyproject_version(new_version)

if __name__ == "__main__":
    main()
