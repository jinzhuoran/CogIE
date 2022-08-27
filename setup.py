from setuptools import setup

with open("README.md","r",encoding="utf-8") as f:
    readme = f.read()

with open('LICENSE', encoding='utf-8') as f:
    license = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    reqs = f.read()

setup(
    name="cogie",
    version="0.1.4",
    description="CogIE: An Information Extraction Toolkit for Bridging Text and CogNet",
    # long_description_contenxt_type="text/markdown",
    # long_description=readme,
    url="https://github.com/jinzhuoran/CogIE/",
    author="CogNLP Team",
    author_email="zhuoran.jin@nlpr.ia.ac.cn",
    license='Apache',
    install_requires=reqs.strip().split("\n"),
    python_requires=">=3.7,<3.9",
    zip_safe=False,
)