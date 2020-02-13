# MachineLearning

## Prerequsites:

- install Python 3.6.8 (or later)

- set env vars

  $ set PATH=C:\Users\developer\AppData\Local\Programs\Python\Python36;%PATH%  
  $ set PYTHONPATH=%PYTHONPATH%

- check pip

  $ pip --version

- install virtualenv via cmd

  $ pip install --user pipenv

## Common tasks:

Prerequsites: VSCode, terminal wirndow, power shell

- create environment p362tfgpu

  virtualenv -p C:\Users\developer\AppData\Local\Programs\Python\Python36\python.exe C:\Projects\virtualenv\p362tfgpu
  
  or
  
  cd C:\Projects\virtualenv  
  python -m virtualenv p362tfgpu

- set access rights

  Set-ExecutionPolicy Unrestricted -Scope CurrentUser

- run activate ps1 script

  for ex:
  C:\Projects\virtualenv\p362tfgpu\Scripts\activate.ps1

- check python version 

  python --version

- install tensorflow (https://www.tensorflow.org/install/install_windows)

  pip3 install --upgrade tensorflow-gpu

- install jupyter notebook

  pip install jupyter

- install scikit-learn

  pip install scikit-learn
  pip install scipy

- install matplotlib

  pip install matplotlib

### download kaggle dataset

- install kaggle (https://github.com/Kaggle/kaggle-api)

  pip install kaggle

- download dataset

  kaggle competitions download -p ../../input/titanic -c titanic

### set path variable

- linux bash

<<<<<<< HEAD
   export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer
   then use $MOD_OPT/mo.py
=======
  export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer
  then use $MOD_OPT/mo.py
  
### jupiter notebook

- install package

  import sys
  !{sys.executable} -m pip install pymc3
>>>>>>> 2c1f7cc77dc8745b3c7371fcf9a89330601d8ecd
