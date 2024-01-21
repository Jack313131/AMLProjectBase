import pkg_resources
import subprocess
import os
import zipfile
import json
import importlib


def save_credentials_cityscapes_dataset():
    filepath = "/root/.local/share/cityscapesscripts/credentials.json"
    credentials = {
        'username': "s285891@studenti.polito.it",
        'password': "AMLProject.2324"
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as file:
        json.dump(credentials, file)


def is_installed(package):
    try:
        pkg_resources.get_distribution(package['name'])
        return True
    except pkg_resources.DistributionNotFound:
        return False


def install_package(package):
    print(f"Installing the package {package['name']}")
    subprocess.check_call(package['command'], shell=True)


# Lista dei pacchetti e comandi di installazione
packages = [
    {"name": "baseProject - AnomalySegmentation_CourseProjectBaseCode",
     "command": "git clone https://github.com/shyam671/AnomalySegmentation_CourseProjectBaseCode.git", "git": True,
     "shell": False},
    {"name": "csDownload", "command": "python -m pip install cityscapesscripts", "git": False, "shell": True},
    {"name": "torch", "command": "pip3 install torch torchvision torchaudio", "git": False, "shell": True},
    {"name": "tqdm", "command": "pip3 install tqdm", "git": False, "shell": True}
]

zipFile = ["gtFine_trainvaltest.zip", "leftImg8bit_trainvaltest.zip", "gtCoarse.zip"]


def data_loader(zip_file_path, extracted_folder_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder_path)


def run_command(command):
    print(f"Running the command :  {command} ...")
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        print("Command execute correctly ...")
        return True, output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Command NOT executed correctly. Error : {e.output.decode('utf-8')}")
        return False, e.output.decode('utf-8')


def git_clone(package):
    print(f"Cloning the repository {package['name']}...")
    try:
        output = subprocess.check_output(package['command'], shell=True, stderr=subprocess.STDOUT)
        print("Repository cloned correctly ...")
    except subprocess.CalledProcessError as e:
        print(f"Repository cloned correctly .... Error : {e.output.decode('utf-8')}")


if __name__ == '__main__':
    print("Start preparing the environment ...")
    rootPath = 'content'
    os.chdir(rootPath)
    os.environ['CITYSCAPES_DATASET'] = f"/{rootPath}/AnomalySegmentation_CourseProjectBaseCode/"
    extracted_folder_path = f'/{rootPath}/AnomalySegmentation_CourseProjectBaseCode/'
    save_credentials_cityscapes_dataset()
    for package in packages:
        if package['git'] == True and os.path.isdir(f"/{rootPath}/AnomalySegmentation_CourseProjectBaseCode") == False:
            git_clone(package)
        elif package['git'] == True:
            print(f"Repository {package['name']} already cloned")
        if package['shell'] == True and not is_installed(package):
            install_package(package)
        else:
            print(f"Package {package['name']} is already installed")

    for zip in zipFile:
        run_command(f"csDownload {zip}")
        data_loader(f"/{rootPath}/{zip}", extracted_folder_path=extracted_folder_path)

    createTrainIdLabelImgs_module = importlib.import_module("cityscapesscripts.preparation.createTrainIdLabelImgs")
    createTrainIdLabelImgs = createTrainIdLabelImgs_module.main
    createTrainIdLabelImgs()

    print("Environment ready  ...")

