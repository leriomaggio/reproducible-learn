# DAP: Data Analysis Protocol Framework

This module implements the *Data Analysis Protocol* as a standalone Python module 
named `dap`.

In this way, one can easily integrate the DAP into data science projects at the cost of 
integrating a new Python module dependency.

__TODO__: Improve and elaborate this!!

## Get the Code

    git clone https://gitlab.fbk.eu/MPBA/dap.git

## Requirements

The `dap` module strongly depends on the following Python packages:

* numpy
* scikit-learn
* pandas
* Keras (for deep learning)
* mlpy

The complete set of dependencies are provided in `requirements.tx`:

```
pip install -r requirements.txt
```

### Notes about `mlpy` dependency

The `mlpy` package available on PyPI is outdated and not working on OSX platforms.
However it is possible to install `mlpy` dependency from the Gitlab repository  :

	```
    pip install git+https://gitlab.fbk.eu/MPBA/mlpy.git
    ```

More information on mlpy installation can be found [here](https://gitlab.fbk.eu/MPBA/mlpy.git/README.md)

#### Test the Installation

To verify that `mlpy` package has been properly installed, type the following command in a terminal 

	```
        python -c "import mlpy; print(mlpy.__version__);"
	```


