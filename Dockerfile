FROM continuumio/miniconda3:latest

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN apt-get update && apt-get install gcc python-dev texlive-xetex -y
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "read", "/bin/bash", "-c"]

# The code to run when container is started:
COPY main.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "read", "python", "main.py"]
