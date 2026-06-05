## DEPLOIEMENT DE K8S

```bash
minikube start --driver=docker

kubectl get nodes

minikube dashboard
```

## LANCER L'ENTRAINEMENT DU MODELE VIA RAYCLUSTER 

Depuis AIA/Projet-final/k8s/ray_cluster :

```bash
kubectl get services

kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265
```

Depuis un nouveau terminal dans AIA/Projet-final/k8s/ray_cluster : 

```bash
ray job submit --runtime-env=runtime.yaml --address="http://127.0.0.1:8265" -- python train_with_ray.py
```


## POUR LANCER AIFLOW 

Depuis AIA/Projet-final/airflow :

```bash
docker-compose up -d
```

http://localhost:8081


## BASE DE DONNEE NEONDB 

https://console.neon.tech/app/projects/shiny-brook-69726525/branches/br-cool-union-ami7s2bj/tables?database=neondb