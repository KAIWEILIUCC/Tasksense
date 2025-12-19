import os


def print_tree(current_dir, indent="", last=True, excluded_ext=[], excluded_dirs=[], non_recursive_dirs=[]):
    try:
        entries = sorted(os.listdir(current_dir))
    except PermissionError:
        print(indent + '└── [Permission Denied]')
        return

    entries = [e for e in entries if not e.startswith('.')]
    dirs = []
    files = []
    for e in entries:
        path = os.path.join(current_dir, e)
        if os.path.isdir(path):
            if e not in excluded_dirs:
                dirs.append(e)
        else:
            if not any(e.endswith(ext) for ext in excluded_ext):
                files.append(e)
    entries = dirs + files

    for index, entry in enumerate(entries):
        path = os.path.join(current_dir, entry)
        is_last = index == len(entries) - 1
        connector = "└── " if is_last else "├── "
        print(indent + connector + entry)
        if os.path.isdir(path):
            if entry in non_recursive_dirs:
                continue 
            else:
                new_indent = indent + ("    " if is_last else "│   ")
                print_tree(path, indent=new_indent, last=is_last, excluded_ext=excluded_ext, excluded_dirs=excluded_dirs, non_recursive_dirs=non_recursive_dirs)

if __name__ == "__main__":
    exclude_extensions = ['.pyc']
    exclude_dirs = ['__pycache__']
    non_recursive_dirs = ['model_zoo', 'sensor_database']
    start_directory = "/aiot-nvme-15T-x2-hk01/home/kaiwei/huawei_project/TaskSense/src"
    print(start_directory)
    print_tree(start_directory, excluded_ext=exclude_extensions, excluded_dirs=exclude_dirs, non_recursive_dirs=non_recursive_dirs)