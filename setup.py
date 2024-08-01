from setuptools import find_packages,setup
from typing import List

HP='-e .'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]
        if HP in requirements:
            requirements.remove(HP)

    return requirements


setup(
 name='mlproject',
 version='0.0.1',
 author='Aaditya surana',
 author_email='aadisurana009@gmail.com',
 packages=find_packages(),
 install_requires=get_requirements('requirements.txt')

)