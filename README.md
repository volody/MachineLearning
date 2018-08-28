# MachineLearning

## Common tasks:

Prerequsites: VSCode, terminal wirndow, power shell

- create environment p362tfgpu

  virtualenv -p C:\Users\developer\AppData\Local\Programs\Python\Python36\python.exe C:\Projects\virtualenv\p362tfgpu

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