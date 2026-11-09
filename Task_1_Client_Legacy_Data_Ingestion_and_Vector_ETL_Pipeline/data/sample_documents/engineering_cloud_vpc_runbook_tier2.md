# Enterprise Multi-VPC Architecture Runbook
**Document ID:** ENG-RUN-2026-08  
**Department:** Engineering  
**Clearance Level:** Tier 2 (Internal)  
**Last Updated:** 2026-08-10  

## 1. Network Topology & Peering
The core infrastructure is segmented into three dedicated Virtual Private Clouds (VPCs):
- pc-prod-compute (CIDR: 10.100.0.0/16): Houses Kubernetes clusters and FastAPI worker nodes.
- pc-prod-data (CIDR: 10.200.0.0/16): Isolated subnet for PostgreSQL read-replicas and ChromaDB vector persistence.
- pc-corp-transit (CIDR: 10.50.0.0/16): Central ingress with AWS Transit Gateway and WAF filtering.

## 2. Transit Gateway Configuration & Routing
Transit Gateway attachments interconnect pc-prod-compute with pc-prod-data across multi-AZ subnets.
Direct egress to the public internet is strictly prohibited for pc-prod-data.
All internal microservice calls must route via AWS PrivateLink endpoints.

## 3. Disaster Recovery & Failover SLAs
- Recovery Time Objective (RTO): Under 15 minutes via automated Cross-Region Route53 health checks.
- Recovery Point Objective (RPO): Under 60 seconds using continuous WAL streaming for PostgreSQL and automated EBS snapshot sync.
