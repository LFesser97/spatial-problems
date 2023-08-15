# License

This work is released as free software under the GNU Public License.  All constituent source code files are covered by this license.  See the file COPYING for the full legal details of the license.

# Usage

To use this software, first set up a python environment with its dependencies.  The environment.yml file describes these dependencies, and if you're using the conda environment management tool or one of its offshoots, can be used to set up the environment appropriately.

For all scripts, run with `-h` or `--help` for some information on usage and arguments.  Most scripts are configured using the hydra library [https://hydra.cc/], and so the standard hydra CLI allows you to modify their configuration with command-line arguments.

## Training

If you're not using the pre-trained model weights (information on how to get them in the "Model Weights" section), you'll need to train your own model.  To generate a training dataset, use the `simulation/citygraph_dataset.py` script.  Note that right now, there's a bug for running the algorithm on batches of graphs with different numbers of nodes, so you should pass the same value to `--min` and `--max` to make sure all graphs in the dataset have the same size.  The dataset will be output to the directory you specify.

To train a model, use the script `learning/inductive_route_training.py`.  You will need to specify the path to your generated training dataset directory as follows:

```python learning/inductive_route_training.py dataset.kwargs.path=/path/to/your/dataset```

By default, the model will be trained over a range of cost weights.  To train just on an operator perspective setting, add the argument `experiment/cost_function=op`,
or to train on a passenger perspective setting, add `experiment/cost_function=pp`.

Training should take around 1-2 hours on a modern commercial GPU.

You can optionally add the argument `+run_name=my_run_name` to name the training run, which will affect the name of the tensorboard logs (stored by default in a directory called `training_logs`) and the name of the output weight file.  If this is not provided, the current date and time will be used as the name of the run.  

When training is complete, the trained weights will be stored in the directory `output` in a file named `inductive_run-name.pt`.
 
## Evaluation

The Mumford dataset, including Mandl, can be downloaded from https://users.cs.cf.ac.uk/C.L.Mumford/Research%20Topics/UTRP/CEC2013Supp.zip.  Download the archive and extract it somewhere.

To evaluate a model on a Mumford city, use the script `learning/eval_route_generator.py`.  You must provide a `.pt` file with model weights, the path to the `Instances` sub-directory of the mumford dataset, and the name of the city on which to evaluate (`mandl` or `mumford0` - `mumford3`), as follows:

```
python learning/eval_route_generator.py +model.weights=path_to_weights.pt eval.dataset.path=/path/to/mumford/Instances +eval=mandl
```

To run Bee Colony Optimization, the signature is similar, but without model weights:

```
python learning/bee_colony.py eval.dataset.path=/path/to/mumford/Instances +eval=mandl
```

And to run Neural BCO, use the same script but specify the neural_bco_mumford config file and provide model weights:
```
python learning/bee_colony.py --config-name neural_bco_mumford +model.weights=path_to_weights.pt eval.dataset.path=/path/to/mumford/Instances +eval=mandl
```

# Model weights

Model weights used for the ITSC experiments can be downloaded from the following link:
https://www.cim.mcgill.ca/~mrl/projs/transit_learning/

# Testing commit hook

If making changes to this repo, it is recommended to add the following to the [hooks] section of your .hgrc file:
`pre-commit = pytest`
This will ensure that all tests will be run before each commit, and the commit will be prevented if all tests do not pass.

