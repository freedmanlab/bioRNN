# bioRNN

To import these modules from anywhere, run `pip install -e .`

To install all dependencies, do `pip install -r requirements.txt`

To see the model in action, start at `example.ipynb` or `experiments/supervised_example/`. 

Note that the parameters in `xdg_parameters.py` affect only on the data generator `xdg_stimulus.py`, and not the model described. RNN parameters sitting in `xdg_paramters.py` are a remnant of how this file was used in a previous repo.

For model description, see the Methods section of [Circuit mechanisms for the maintenance and manipulation of information in working memory](https://www.nature.com/articles/s41593-019-0414-3).

The model was written as a [Sonnet](https://sonnet.readthedocs.io/en/latest/index.html) module so that you can compose it with other Sonnet modules. I recommend working with Sonnet modules because of their simplicity and readable source code. If you want to use Keras modules, note that Sonnet modules are not always composable with Keras modules. Keras functions are okay though (like `tf.keras.optimizers` and `tf.keras.initializers`). If you want to use Keras modules, you'd probably want to rewrite the model as a Keras module. This should be easy enough, you'll just have to rename a few things (see [this](https://www.tensorflow.org/guide/keras/custom_layers_and_models) and [this](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AbstractRNNCell)).

If you're new to TensorFlow 2, I wrote this [tutorial notebook](https://colab.research.google.com/drive/16Wr40c-iGf-3FL6m0j6Yxnw-ynId-76c?usp=sharing).
