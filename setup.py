import os
import re
from glob import glob
import configparser
from setuptools import find_packages, setup

setup_cfg = configparser.ConfigParser()
setup_cfg.read("setup.cfg")
package_name = setup_cfg['default']['package_name']
version = setup_cfg['default']['version']

data_files = []
for root, dirs, files in os.walk("configuration"):
    data_files.append(
        (os.path.relpath(root, "configuration"), [os.path.join(root, f) for f in files])
    )

# declare your scripts:
# scripts in bin/ with a shebang containing python will be
# recognized automatically
scripts = []
for fname in glob("bin/*"):
    with open(fname, "r") as fh:
        if re.search(r"^#!.*python", fh.readline()):
            scripts.append(fname)

with open("requirements.txt") as f:
    dependencies = [line for line in f]

setup(
    name=package_name,
    version=version,
    description="Slack utilities",
    author="Shern",
    author_email="18212522+shern2@users.noreply.github.com",
    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=dependencies,
    include_package_data=True,
    # include data files
    data_files=data_files,
    test_command="none",
    doc_command="none",
    # Enable build-time format checking
    check_format=False,
    # Enable type checking
    test_mypy=False,
    # Enable linting at build time
    test_flake8=False,
)
