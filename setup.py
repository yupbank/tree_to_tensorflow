from setuptools import setup
from collections import defaultdict
from pip.req import parse_requirements

requirements = []
extras = defaultdict(list)
for r in parse_requirements('requirements.txt', session='hack'):
    if r.markers:
        extras[':' + str(r.markers)].append(str(r.req))
    else:
        requirements.append(str(r.req))

setup(name='TFTree',
      version='0.1.0',
      packages=['ttt'],
          install_requires=requirements,
    extras_require=extras,
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
