FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

WORKDIR /app


COPY requirements.txt /app/


RUN pip install --no-cache-dir -r requirements.txt


COPY . /app

CMD ["uvicorn", "Redemption_Model:app", "--host", "0.0.0.0", "--port", "80"]
