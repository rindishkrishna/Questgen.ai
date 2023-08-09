from distutils.core import setup

setup(name='Questgen',
      version='1.0.1',
      description='Question generator from any text',
      author='Questgen contributors',
      author_email='rindishkrishna@gmail.com',
      packages=['Questgen', 'Questgen.encoding'],
      url="https://github.com/rindishkrishna/Questgen.ai.git",
      install_requires=[
            'torch==2.0.1',
            'transformers==4.29.2',
            'sense2vec==2.0.2',
            'strsim==0.0.3',
            'six==1.16.0',
            'networkx==3.1',
            'numpy==1.22.4',
            'scipy==1.10.1',
            'scikit-learn==1.2.2',
            'unidecode==1.3',
            'future==0.18.3',
            'joblib==1.2.0',
            'pytz==2022.7.1',
            'python-dateutil==2.8.2',
            'flashtext==2.7',
            'pandas==1.5.3',
            'sentencepiece==0.1.99'
      ],
      package_data={'Questgen': ['questgen.py', 'train_gpu.py', 'encoding.py']}
      )
