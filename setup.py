from setuptools import setup
from poetry.core.packages import ProjectPackage

setup(
    name='cassandra',
    version='0.1.0',
    description='Description of your package',
    author='Stefano Tolomeo',
    author_email='stafano.tolomeo91@email.com',
    packages=['cassandra'],
    install_requires=[
        'dependency1',
        'dependency2',
        # Specify any other dependencies your package requires
    ]
)
