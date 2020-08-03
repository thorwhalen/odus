version = '0.0.6'
root_url = 'https://github.com/thorwhalen'

# import os
# name = os.path.split(os.path.dirname(__file__))[-1]

name = 'odus'


def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except:
        return ""


ujoin = lambda *args: '/'.join(args)

if root_url.endswith('/'):
    root_url = root_url[:-1]


def my_setup(print_params=True, **setup_kwargs):
    from setuptools import setup
    if print_params:
        import json
        print("Setup params -------------------------------------------------------")
        print(json.dumps(setup_kwargs, indent=2))
        print("--------------------------------------------------------------------")
    setup(**setup_kwargs)


dflt_kwargs = dict(
    name=f"{name}",
    version=f'{version}',
    url=f"{root_url}/{name}",
    author='Thor Whalen',
    author_email='thorwhalen1@gmail.com',
    # license='Apache Software License',
    license='MIT',
    include_package_data=True,
    platforms='any',
    long_description=readme(),
    long_description_content_type="text/markdown",

)

# setup_kwargs = format_str_vals_of_dict(dflt_kwargs, name=name, root_url=root_url, version=version)
setup_kwargs = dflt_kwargs

more_setup_kwargs = dict(
    install_requires=[
        'py2store',
        'pandas',
        'numpy',
        'Pillow',
        'spyn',
        'matplotlib',
        'openpyxl',  # to do raw data diagnosis (namely to get colors from excel)
        'argh'  # to create nicely-interfaced scripts easily
    ],
    description="Tools to provide easy access to prepared data to data scientists that can't be asked.",
    keywords=['data', 'data access', 'drug use', 'markov', 'bayesian'],
    # download_url='{root_url}/{name}/archive/v{version}.zip'),
)

setup_kwargs = dict(setup_kwargs, **more_setup_kwargs)

my_setup(**setup_kwargs)
