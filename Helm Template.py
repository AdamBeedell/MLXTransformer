import os

# Define the Helm chart directory structure
base_dir = "/mnt/data/my-stack"
dirs = [
    base_dir,
    os.path.join(base_dir, "templates")
]

# Create the directory structure
for d in dirs:
    os.makedirs(d, exist_ok=True)

# Define file contents
chart_yaml = """\
apiVersion: v2
name: my-stack
description: A dev environment with frontend, backend, postgres, and portainer
type: application
version: 0.1.0
"""

values_yaml = """\
frontend:
  image: python:3.10-slim
  command: ["streamlit", "run", "UI.py"]
  port: 8501
  nodePort: 30501

backend:
  image: python:3.10-slim
  command: ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
  port: 8000
  nodePort: 30500

postgres:
  image: postgres:15
  user: user
  password: password
  db: mydb

portainer:
  port: 9000
  nodePort: 30000
"""

frontend_yaml = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
        - name: frontend
          image: {{ .Values.frontend.image }}
          command: {{ toJson .Values.frontend.command }}
          ports:
            - containerPort: {{ .Values.frontend.port }}
---
apiVersion: v1
kind: Service
metadata:
  name: frontend
spec:
  type: NodePort
  selector:
    app: frontend
  ports:
    - port: 80
      targetPort: {{ .Values.frontend.port }}
      nodePort: {{ .Values.frontend.nodePort }}
"""

backend_yaml = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
        - name: backend
          image: {{ .Values.backend.image }}
          command: {{ toJson .Values.backend.command }}
          ports:
            - containerPort: {{ .Values.backend.port }}
---
apiVersion: v1
kind: Service
metadata:
  name: backend
spec:
  type: NodePort
  selector:
    app: backend
  ports:
    - port: 80
      targetPort: {{ .Values.backend.port }}
      nodePort: {{ .Values.backend.nodePort }}
"""

postgres_yaml = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: {{ .Values.postgres.image }}
          env:
            - name: POSTGRES_USER
              value: {{ .Values.postgres.user | quote }}
            - name: POSTGRES_PASSWORD
              value: {{ .Values.postgres.password | quote }}
            - name: POSTGRES_DB
              value: {{ .Values.postgres.db | quote }}
          ports:
            - containerPort: 5432
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
"""

portainer_yaml = """\
apiVersion: v1
kind: ServiceAccount
metadata:
  name: portainer-sa

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: portainer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: portainer
  template:
    metadata:
      labels:
        app: portainer
    spec:
      serviceAccountName: portainer-sa
      containers:
        - name: portainer
          image: portainer/portainer-ce:latest
          args:
            - "--http-enabled"
            - "--no-analytics"
          ports:
            - containerPort: {{ .Values.portainer.port }}
          volumeMounts:
            - name: portainer-data
              mountPath: /data
      volumes:
        - name: portainer-data
          emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: portainer
spec:
  type: NodePort
  selector:
    app: portainer
  ports:
    - port: {{ .Values.portainer.port }}
      targetPort: {{ .Values.portainer.port }}
      nodePort: {{ .Values.portainer.nodePort }}
"""

# Write files
with open(os.path.join(base_dir, "Chart.yaml"), "w") as f:
    f.write(chart_yaml)

with open(os.path.join(base_dir, "values.yaml"), "w") as f:
    f.write(values_yaml)

with open(os.path.join(base_dir, "templates", "frontend.yaml"), "w") as f:
    f.write(frontend_yaml)

with open(os.path.join(base_dir, "templates", "backend.yaml"), "w") as f:
    f.write(backend_yaml)

with open(os.path.join(base_dir, "templates", "postgres.yaml"), "w") as f:
    f.write(postgres_yaml)

with open(os.path.join(base_dir, "templates", "portainer.yaml"), "w") as f:
    f.write(portainer_yaml)

base_dir
