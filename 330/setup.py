from setuptools import setup, find_packages

setup(name='train_for_google',
  version='0.1',
  packages=find_packages(),
  description='example to run keras on gcloud ml-engine',
  author='Seara Chen',
  author_email='searachen@gmail.com',
  license='MIT',
  install_requires=[
      'keras',
      'h5py',
      'sklearn'
      'matplotlib'
  ],
  zip_safe=False)