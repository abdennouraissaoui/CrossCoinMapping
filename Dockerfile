# Use the official Python image with version 3.10.12
FROM python:3.10.12-slim

# Set environment variables
ENV JUPYTER_NO_CONFIG=true
ENV SHELL=/bin/bash

# Install JupyterLab and other common packages
RUN pip install --no-cache-dir jupyterlab

# Set the working directory
WORKDIR /home/jovyan/work

# Expose the default Jupyter port
EXPOSE 8888

# Set the command to start JupyterLab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
