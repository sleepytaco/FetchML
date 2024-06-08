FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install poetry

# Install required dependencies through poetry
RUN poetry install

# Run the application through poetry
ENTRYPOINT ["poetry", "run", "python", "main.py"]
