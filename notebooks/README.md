# Jupyter Notebooks for Exploratory Analysis and Result Reporting

## SSH Colab
`ssh_colab.ipynb` notebook sets up an ssh connection with colab session and we can use GPU's to run experiments. If run for the first time, it will require some configuration, but the steps are described in the cell outputs **(Client machine configuration)**. You also need to create a `git_config.json` file on your Google Drive to store git configuration (email, username, personal_token,branch). Using the `git_config.json` we won't accidentally expose our personal_token to github when we push and it won't revoke it.
