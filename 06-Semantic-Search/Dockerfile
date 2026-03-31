FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY 01-fruits-vector-setup.ipynb 02-fruits-similarity-metrics.ipynb 03-fruits-similarity-student.ipynb guide.md /workspace/

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--LabApp.token=''", "--LabApp.password=''"]
