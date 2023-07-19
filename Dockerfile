FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

WORKDIR /app/src

CMD ["python", "-u", "main.py"]
