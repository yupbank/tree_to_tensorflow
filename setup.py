from setuptools import setup


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(name='ttt',
      version='0.1.0',
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
        keywords=['tensorflow', 'machine learning', 'sklearn', 'spark', 'model-serving'],
        classifiers=[],
)
