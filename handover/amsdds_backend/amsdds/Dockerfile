FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
ENV AMSDDS_WEIGHTS=/weights PORT=8000
EXPOSE 8000
CMD ["python", "app.py"]
# run: docker run --gpus all -p 8000:8000 -v /path/to/weights:/weights amsdds
