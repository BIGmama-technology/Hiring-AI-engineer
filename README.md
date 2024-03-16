# AI engineer test

The principal objective of this project is to evaluate the applicant's ability to learn new skills on the fly, build machine learning models in adherence to best practices and colaborate with others.

The applicant is also expected to write a modular code following good coding practices.

## How does this work ?

Below is a list of tasks that candidates concurently work on. If you deem your contribution to be complete, you can create a pull request.
The team will review your contribution and provide feedback. If its good your branch will be merged with the `main` branch.
Tasks that are done will be ommited and new tasks will apear for others.
Candidates with merged changes will be invited to pass an interview with the team.

## Who can apply ?

Both students looking for an internship at BIGmama and professionals looking for a full-time position can apply.

## Tasks

- [x] **GaussianProcess.py**: Write a `GaussianProcess` class that embodies the Gaussian process regression model's functionality.
- [x] **kernels.py**: Implement a selection of three kernel functions.
- [x] **Kernel Operations**: Enable your kernels to perform addition (`+`) and multiplication (`*`) operations.
- [x] **Fit the guassian process**: Fit your Gaussian process to the datasets provided and plot the results.
- [x] **Optimize gaussian process fit fucntion**: for loops are slow, try to optimize the fit function to be faster.
- [x] **Add 2 periodic kernels**: Add 2 periodic kernels to `kernels.py`.

---

- [x] **simple BNN** : implement a bayesian neural network using `pytorch` or `pymc3`.
- [x] **fit BNN** : fit bnn to provided data and generate plots.
- [ ] **improve BNN results** : improve the architecture.
- [ ] **improve training loop** : refactor + use `tqdm`
- [ ] benchmark current models using relavant metrics and datasets of your choice (give clear instructions in the README on how to benchmark future added models)
- [ ] <span style="color: green">**Variational Inference** : implement variational inference for BNNs.</span> (extra love if you do this.)
- [ ] **plot epistmic uncertainty** : BNNs like GPs allow us to quantify uncertainty.

---

- [ ] **simple MLP** : impelement a simple MLP with batch normalization and dropout for modeling uncertainty [paper 1](https://arxiv.org/abs/1802.06455) [paper 2](https://arxiv.org/abs/1506.02142)
- [ ] **visualize uncertainty** : visualize the modeled uncertainty using the provided datasets

---

- [ ] **Github actions** : improve developer experience with github actions (start with tests).
- [ ] **write tests** : use pytest to test GP and kernels modules.

---

- [ ] **Inference with GPs** : plot how the GP fit the porvided data using different kernels, plot uncertainty too.
- [ ] **Generalize**: so we can run gaussian process on any dataset, not just the ones provided.
- [ ] **REST API via FastAPI**: Design a REST API using FastAPI to make your Gaussian process regression accessible over HTTP.

---

- [ ] **Build a user interface**: Build a user interface to interact with the gaussian process model.
- [ ] **Dockerization**: Containerize your application with Docker, ensuring all dependencies are included for seamless setup and deployment.
- [ ] **Refactor**: Refactor code following good practices and a design pattern of your choice.
- [ ] **Documentation**: Document the project thoroughly with docstrings, inline comments and using a documentation generator of your choice.

## Setup

Clone the repository

```bash
git clone git@github.com:BIGmama-technology/Hiring-AI-engineer.git
```

Run `setup.sh`, this will create a virtual environment and install some dependencies

```bash
./scripts/setup.sh
```

Activate the virtual environment

```bash
source .venv/bin/activate
```

To train BNN run :

```bash
python src/main.py
```

To run the server run :

```bash
uvicorn src.api.app:app --reload
```

## Contribution guidelines

- design the structure of your repo in a modular way, example :

```
.
├── data
│   ├── international-airline-passengers.csv
│   └── mauna_loa_atmospheric_co2.csv
├── docs
│   └── report.pdf
├── LICENSE
├── output
│   └── figure_1.png
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── data
│   │   └── data_loader.py
│   ├── models
│   │   ├── GaussianProcess.py
│   │   └── kernels.py
│   └── utils
│       └── utils.py
├── pyproject.toml
├── README.md
└── setup.cfg
```

- always use the virtual environment

```bash
# activate the virtual environment created by setup.sh
source .venv/bin/activate
```

- Make sure you include any requirements and dependencies in your `pyproject.toml` or `requirements.txt`.
- Type your code, document it and format it.

```python
# untyped, undocumented and unformatted code
import numpy as np
class gaussiankernel:
 def __init__(self,sigma=1.0):
  self.sigma=sigma
 def compute(self,x1,x2):
  return np.exp(-0.5 * np.linalg.norm(x1-x2)**2 / self.sigma**2)

```

```python
# typed, documented and formatted code
import numpy as np
from typing import Any, Union

class GaussianKernel:
    def __init__(self, sigma: float = 1.0) -> None:
        """
        Initialize the Gaussian kernel with a specified standard deviation (sigma).

        Parameters:
        sigma (float): The standard deviation of the Gaussian kernel.
        """
        self.sigma: float = sigma

    def compute(self, x1: Union[float, np.ndarray], x2: Union[float, np.ndarray]) -> Any:
        """
        Compute the Gaussian kernel between two points.

        Parameters:
        x1 (Union[float, np.ndarray]): The first point or vector.
        x2 (Union[float, np.ndarray]): The second point or vector.

        Returns:
        The computed Gaussian kernel value.
        """
        return np.exp(-0.5 * np.linalg.norm(x1 - x2) ** 2 / self.sigma ** 2)

```

- Commit often and write meaningful commit messages.
- Create a new branch with your name, push your code to it and create a pull request once you finish your contribution.

## Resources

Candidates should leverage the following resources for guidance:

- [Good practices](https://goodresearch.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Intro to Docker](https://docker-curriculum.com/)
- [What are gaussian processes : interactive guide](https://distill.pub/2019/visual-exploration-gaussian-processes/)
- [Kernel cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/)
- [Packaging with pip](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Varitational inference](https://krasserm.github.io/2019/03/14/bayesian-neural-networks/)

## FAQ

#### how many features should I work on ?

doesn't matter, what important is the value of your contribution and it's quality, impress us !

#### what if the task I am working on gets completed by someone else ?

pick another task, and hurry up !

#### I dont arrive to push my code !
here is an issue with details on how to push [#21](https://github.com/BIGmama-technology/Hiring-AI-engineer/issues/21)

#### what if I have a question ?

open an issue and we will answer it as soon as possible !

btawfiq inchalah
