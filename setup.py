from setuptools import setup


requirements = [
    'tensorflow<=1.15.0',
    'scipy>=0.17',
    'numpy>=1.10',
    'scikit-learn>=0.22'
]

setup(name='TFTree',
      version='0.1.1',
      packages=['ttt'],
      install_requires=requirements,
      description='Tree to tensorflow',
      long_description="""
        ttt is a machine learning toolkit designed to efficiently perform
        tree model exporting into tensorflow estimators.
        """,
      author='Peng Yu',
      author_email='yupbank@gmail.com',
      url='https://github.com/yupbank/tree_to_tensorflow',
      download_url='https://github.com/yupbank/tree_to_tensorflow/archive/0.1.0.tar.gz',
      keywords=['tensorflow', 'machine learning',
                'sklearn', 'spark', 'model-serving'],
      classifiers=[],
      )
