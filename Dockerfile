FROM python:3.10
ADD . .
RUN pip install --upgrade pip
RUN pip install -r requirement.txt
EXPOSE 8001
CMD ["python","app.py"]
