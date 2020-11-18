import os
from os.path import isfile, join
import subprocess
from glob import glob

path = '.'
# get all python files in the directory
py_files = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.py'))]


for f in py_files:
    print(f'{f} issues:')
    output = subprocess.run(['pycodestyle', f, '--hang-closing'],
                            capture_output=True, text=True)
    print(output.stdout)
    assert len(output.stdout) == 0
    assert output.returncode == 0

print('No issues!')
