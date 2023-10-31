from setuptools import find_packages, setup

install_requires = ["pandas", "torch"]

setup(
    name="prediction_client",
    version="0.0.1",
    packages=find_packages(),
    data_files=[],
    install_requires=install_requires,
    include_package_data=False,
)
